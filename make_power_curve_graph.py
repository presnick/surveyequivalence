def make_power_curve_graph(expert_scores, amateur_scores ,classifier_scores, points_to_show_surveyEquiv=None):

    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    ax = fig.add_subplot(111)

    # If there are expert_scores and show lines is false:
    if expert_scores and not expert_scores['Show_lines']:
            x = list(expert_scores['Power_curve']['k'])
            y = list(expert_scores['Power_curve']['score'])
            yerr = list(expert_scores['Power_curve']['confidence_radius'])

            ax.errorbar(x, y, yerr=yerr, marker='o',color = expert_scores['color'], elinewidth = 2, capsize = 5,label=expert_scores['name'], linestyle='')


    # If there are expert_scores and show_lines is true:
    if expert_scores and expert_scores['Show_lines']:
        x = list(expert_scores['Power_curve']['k'])
        y = list(expert_scores['Power_curve']['score'])
        yerr = list(expert_scores['Power_curve']['confidence_radius'])

        ax.errorbar(x, y, yerr=yerr, marker='o',color = expert_scores['color'], elinewidth = 2, capsize = 5,label=expert_scores['name'], linestyle='-')



    # If there are amateur_scores show_lines is false
    if amateur_scores and not amateur_scores[0]['Show_lines']:
        x=[]
        y=[]
        yerr=[]

        for i in (range(len(amateur_scores))):
            x.append(list(amateur_scores[i]['Power_curve']['k']))
            y.append(list(amateur_scores[i]['Power_curve']['score']))
            yerr.append(list(amateur_scores[i]['Power_curve']['confidence_radius']))

        for i in range(len(amateur_scores)):
            ax.errorbar(x[i], y[i], yerr=yerr[i], marker='o',color = amateur_scores[i]['color'], label=amateur_scores[i]['name'],elinewidth = 2, capsize = 5, linestyle='')



    # If there are amateur_scores and show_lines is true:
    if amateur_scores and amateur_scores[0]['Show_lines']:
        x=[]
        y=[]
        yerr=[]

        for i in (range(len(amateur_scores))):
            x.append(list(amateur_scores[i]['Power_curve']['k']))
            y.append(list(amateur_scores[i]['Power_curve']['score']))
            yerr.append(list(amateur_scores[i]['Power_curve']['confidence_radius']))

        for i in range(len(amateur_scores)):
             ax.errorbar(x[i] , y[i],yerr=yerr[i],linestyle='-',marker='o',color = amateur_scores[i]['color'],label=amateur_scores[i]['name'],elinewidth = 2, capsize = 5)


    #if classifier_scores has a confidence interval:
    if classifier_scores['confidence_radius'].empty is False:
        ci=[float(i) for i in classifier_scores['confidence_radius'].to_list()]
        for (i, score) in enumerate(classifier_scores['scores']):
            y_=float(classifier_scores['scores'].iloc[i])
            ax.axhline(y=y_,  color=classifier_scores['colors'].iloc[i],linewidth=2, linestyle='dashed',label=classifier_scores['names'].iloc[i])
            ax.axhspan(y_ - ci[i],y_+ ci[i], alpha=0.5, color=classifier_scores['colors'].iloc[i])



    #if classifier_scores has no confidence interval:
    if classifier_scores['confidence_radius'].empty:
        # if there are Classifier_scores:
        if classifier_scores['scores'].empty is False:
            for (i, score) in enumerate(classifier_scores['scores']):
                axhline(y=score,  color=classifier_scores['colors'].iloc[i],linewidth=2, linestyle='dashed',label=classifier_scores['names'].iloc[i])


    #If  Points_to_show_SurveyEquiv exists:
    if points_to_show_surveyEquiv is not None:


        expert_scores_copy = expert_scores['Power_curve'].copy()
        f = expert_scores_copy['k']==0

        expert_scores_copy.where(f, inplace = True)

        expert_score_at_0 = expert_scores_copy.dropna()['score'].iloc[0]

    #if  (score, which is the y value at point x)<expert score at 0 return 0
    #[does this means plot at point 0,0?]-------------------------------
        x_intercepts=[0,54]

        for i in range(len(se)):
                #else: get min(k:where expert score at k>our score)

                #if expert line never above our score, return 1> maximum number of expert raters.

                if se[i]['which_type'] == 'classifier':

                    classifier_copy = classifier_scores.copy()
                    f = classifier_copy['names']==se[i]['name']
                    classifier_copy.where(f, inplace = True)
                    y= float( classifier_copy.dropna()['scores'].iloc[0])
                    x_intercept = np.interp(y, expert_scores['Power_curve']['score'].to_list(), expert_scores['Power_curve']['k'].to_list())

                    x_intercepts.append(x_intercept)
                    plt.scatter(x_intercept, y,c='black')
                    plt.axvline(x=x_intercept,  color='black',linewidth=2, linestyle='dashed',ymax =y)


                if se[i]['which_type'] == 'amateur':

                    for j in amateur_scores:
                        if se[i]['name'] == j['name']:
                            #find the points that make up the line
                            x = j['Power_curve']['k'].tolist()
                            y = j['Power_curve']['score'].tolist()
                            #given x_value, find the corresponding y value for that point on the line
                            y_intercept_at_x_value= np.interp(se[i]['which_x_value'], x,y)
#                             if (y_intercept_at_x_value<expert_score_at_0):
#                                 print('y_intercept_at_x_value<expert_score_at_0')

                            x_intercept = np.interp(y_intercept_at_x_value, expert_scores['Power_curve']['score'].to_list(), expert_scores['Power_curve']['k'].to_list())
                            x_intercepts.append(x_intercept)
                            plt.scatter(x_intercept, y_intercept_at_x_value,c='black')
                            plt.axvline(x=x_intercept,  color='black',linewidth=2, linestyle='dashed',ymax =y_intercept_at_x_value)
        x_intercepts.sort()
        plt.xticks([i for i in x_intercepts])
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))






    ax.axis([0, 54, 0, 1])
    ax.set_xlabel('Number of other journalists', fontsize = 16)
    ax.set_ylabel('Correlation with reference journalist', fontsize = 16)
    plt.legend(loc='upper right')
    plt.show()
