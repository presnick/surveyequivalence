"""Collect a frozen, public Kalshi NYC weather sample without fitting models.

These aggregate quotes do not provide the individual responses required for
the professor's survey-type equivalence design. Retained as a data-access probe.

Fetch missing public responses:
    .venv/bin/python research/kalshi_data.py --fetch
Rebuild the export using only the retained cache:
    .venv/bin/python research/kalshi_data.py

All dates and the quote time were fixed before model evaluation. The collector
keeps failed requests and market metadata even when a quote is unavailable.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time as datetime_time, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache/kalshi-question-pilot'
OUTPUT = CACHE / 'source_data.json'
BASE = 'https://external-api.kalshi.com/trade-api/v2'
SERIES = 'KXHIGHNY'
START, END = date(2026, 3, 1), date(2026, 9, 20)
NY = ZoneInfo('America/New_York')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


class PublicClient:
    """At most four concurrent reads, starts spaced at least 0.18 seconds."""

    def __init__(self):
        self.lock = threading.Lock()
        self.next_request = 0.0

    def get(self, url, raw_path):
        attempts = []
        for attempt in range(3):
            with self.lock:
                wait = max(0, self.next_request - time.monotonic())
                if wait:
                    time.sleep(wait)
                self.next_request = time.monotonic() + 0.18
            stamp = utc_now()
            status, body, failure = None, b'', None
            retry_after = None
            try:
                request = urllib.request.Request(
                    url, headers={'Accept': 'application/json',
                                  'User-Agent': 'surveyequivalence-research/1.0'})
                with urllib.request.urlopen(request, timeout=20) as response:
                    status, body = response.status, response.read()
                json.loads(body)
            except urllib.error.HTTPError as error:
                status, body = error.code, error.read()
                retry_after = error.headers.get('Retry-After')
                failure = str(error)
            except Exception as error:
                failure = '{}: {}'.format(type(error).__name__, error)
            route_id = hashlib.sha256((url + stamp).encode()).hexdigest()[:12]
            retained = raw_path if failure is None else raw_path.with_name(
                '{}.{}.attempt{}.error{}'.format(raw_path.stem, route_id,
                                                attempt + 1, raw_path.suffix))
            retained.parent.mkdir(parents=True, exist_ok=True)
            retained.write_bytes(body)
            record = dict(method='GET', url=url, fetched_at_utc=stamp,
                          status=status, sha256=hashlib.sha256(body).hexdigest(),
                          raw_path=str(retained.relative_to(ROOT)),
                          attempt=attempt + 1, error=failure)
            attempts.append(record)
            if failure is None:
                return json.loads(body), attempts
            if status is not None and status != 429 and status < 500:
                break
            if attempt < 2:
                try:
                    pause = float(retry_after) if retry_after else 2 ** (attempt + 1)
                except ValueError:
                    pause = 2 ** (attempt + 1)
                time.sleep(min(15, max(1, pause)))
        return None, attempts


def market_date(ticker):
    return datetime.strptime(ticker.split('-')[1], '%y%b%d').date()


def metadata(client, fetch):
    markets, records, errors = {}, [], []
    # Archived listings take precedence when both routes list the same ticker.
    # The candle request can try the other tier on a 404.
    for tier in ('live', 'historical'):
        cursor, page = '', 1
        while True:
            query = {'series_ticker': SERIES}
            if tier == 'live':
                query['status'] = 'settled'
            query['limit'] = 1000
            if cursor:
                query['cursor'] = cursor
            route = '/markets' if tier == 'live' else '/historical/markets'
            url = BASE + route + '?' + urllib.parse.urlencode(query)
            path = CACHE / 'probe' / 'ny_{}_page{}.json'.format(tier, page)
            sidecar = path.with_suffix('.request.json')
            if path.exists():
                body = path.read_bytes()
                data = json.loads(body)
                if sidecar.exists():
                    records.extend(json.loads(sidecar.read_text()))
                else:
                    # Initial read-only coverage probes predated this collector.
                    records.append(dict(method='GET', url=url, status=200,
                        fetched_at_utc=None,
                        cache_file_modified_at_utc=datetime.fromtimestamp(
                            path.stat().st_mtime, timezone.utc).isoformat(),
                        timestamp_note='Initial coverage probe did not record request time; file mtime retained.',
                        sha256=hashlib.sha256(body).hexdigest(),
                        raw_path=str(path.relative_to(ROOT)), error=None))
            elif fetch:
                data, attempts = client.get(url, path)
                save_json(sidecar, attempts)
                records.extend(attempts)
            else:
                data = None
            if data is None:
                errors.append(dict(kind='metadata_unavailable', tier=tier,
                                   url=url, cache_path=str(path.relative_to(ROOT))))
                break
            page_dates = []
            for source in data.get('markets', []):
                day = market_date(source['event_ticker'])
                page_dates.append(day)
                if not START <= day <= END:
                    continue
                market = dict(source)
                old = markets.get(market['ticker'], {})
                market['_date'] = day.isoformat()
                market['_tier'] = tier
                market['_listing_tiers'] = old.get('_listing_tiers', []) + [tier]
                market['_cutoff_ts'] = int(datetime.combine(
                    day, datetime_time.min, NY).timestamp())
                markets[market['ticker']] = market
            cursor = data.get('cursor', '')
            # Listings are descending by settlement time. The first historical
            # page already covers through February; no older page is needed.
            if not cursor or (page_dates and max(page_dates) < START):
                break
            if page_dates and min(page_dates) < START:
                break
            page += 1
    return sorted(markets.values(), key=lambda m: (m['_date'], m['ticker'])), records, errors


def one_market(market, client, fetch):
    ticker = market['ticker']
    path = CACHE / 'candles' / (ticker + '.json')
    sidecar = CACHE / 'candles' / (ticker + '.request.json')
    if path.exists() and sidecar.exists():
        provenance = json.loads(sidecar.read_text())
        return ticker, json.loads(path.read_text()), provenance['requests'], provenance['errors']
    if sidecar.exists() and not fetch:
        provenance = json.loads(sidecar.read_text())
        return ticker, None, provenance['requests'], provenance['errors']
    if not fetch:
        return ticker, None, [], [dict(kind='candle_cache_missing', ticker=ticker)]
    previous = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    requests = previous.get('requests', [])
    errors = previous.get('errors', [])
    cutoff = market['_cutoff_ts']
    query = urllib.parse.urlencode(dict(start_ts=cutoff - 3 * 3600,
                                       end_ts=cutoff, period_interval=60))
    tiers = [market['_tier'], 'live' if market['_tier'] == 'historical' else 'historical']
    data = None
    for tier in tiers:
        route = ('/historical/markets/{}/candlesticks'.format(ticker)
                 if tier == 'historical' else
                 '/series/{}/markets/{}/candlesticks'.format(SERIES, ticker))
        url = BASE + route + '?' + query
        data, attempts = client.get(url, path)
        for record in attempts:
            record['ticker'] = ticker
            record['tier'] = tier
        requests.extend(attempts)
        errors.extend(dict(kind='request_failed', ticker=ticker,
                           url=record['url'], status=record['status'],
                           error=record['error']) for record in attempts if record['error'])
        if data is not None:
            break
        if not attempts or attempts[-1]['status'] != 404:
            break
    for error in errors:
        error['recovered'] = data is not None
    if data is None:
        errors.append(dict(kind='candle_unavailable', ticker=ticker, recovered=False))
    else:
        # Do not discard absent/null fields or candles based on price/outcome.
        future = [c for c in data.get('candlesticks', []) if c['end_period_ts'] > cutoff]
        if future:
            errors.append(dict(kind='api_returned_after_cutoff', ticker=ticker,
                               count=len(future), recovered=False))
    save_json(sidecar, dict(requests=requests, errors=errors))
    return ticker, data, requests, errors


def collect(fetch=False):
    started = utc_now()
    client = PublicClient()
    markets, requests, errors = metadata(client, fetch)
    candles = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(one_market, market, client, fetch)
                   for market in markets]
        for count, future in enumerate(as_completed(futures), 1):
            ticker, data, records, failures = future.result()
            if data is not None:
                candles[ticker] = data
            requests.extend(records)
            errors.extend(failures)
            if count % 100 == 0 or count == len(markets):
                print('Processed {}/{} contracts; {} cached responses; {} error records'.format(
                    count, len(markets), len(candles), len(errors)), flush=True)
    days = sorted({market['_date'] for market in markets})
    export = dict(markets=markets, candles=candles,
        requests=sorted(requests, key=lambda r: (r['url'], r.get('fetched_at_utc') or '')),
        errors=sorted(errors, key=lambda e: (e.get('ticker', ''), e['kind'])),
        collection=dict(series_ticker=SERIES, start_date=START.isoformat(),
            end_date=END.isoformat(), quote_time='00:00:00 America/New_York at start of outcome date',
            candle_window_hours=3, period_interval_minutes=60,
            started_at_utc=started, completed_at_utc=utc_now(),
            public_read_only=True, network_enabled=fetch,
            events=len(days), contracts=len(markets), candle_responses=len(candles),
            error_records=len(errors), metadata_date_range=[days[0], days[-1]] if days else [],
            metadata_selection='Fixed date window and series; ticker deduplication; no performance or outcome-based filtering.',
            cutoff_rule='Retain API candles requested through midnight; downstream analysis must enforce end_period_ts <= market._cutoff_ts.',
            metadata_probe_note='Original metadata probes retained raw responses and file mtime, but not exact HTTP request timestamps.'))
    save_json(OUTPUT, export)
    print(json.dumps(export['collection'], indent=2), flush=True)
    print('Saved {}'.format(OUTPUT), flush=True)
    return export


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true', help='Fetch uncached public market data.')
    arguments = parser.parse_args()
    collect(fetch=arguments.fetch)
