def make_power_curve(num_raters,confidence_radius=True):
    k = np.array(np.sort(np.append(random.sample([i for i in range(1,54)],num_raters-1),[0]))).reshape(num_raters,1)
    score =np.array(np.sort(random.sample(list(np.random.uniform(0.05, 0.99,15)),num_raters))).reshape(num_raters,1)

    if (confidence_radius):
        Confidence_radius = np.array(np.sort(random.sample(list(np.random.uniform(0, 0.07,10)),num_raters))).reshape(num_raters,1)

    power_curve = np.hstack((k, score,Confidence_radius))
    dataframe=pd.DataFrame(power_curve, columns=['k','score','confidence_radius'])

    return dataframe



def make_items_expert_amateur(base_name,colors,num_raters,expert=False,show_lines=False):
    if (not expert):
        items =[]

    for (i, color) in enumerate(colors):
        name=base_name
        if (not expert):

            name= base_name+str(i)
            items.append({'name':name,'color':color,'Show_lines':show_lines,'Power_curve':make_power_curve(num_raters)})
        else:
            items ={'name':name,'color':color,'Show_lines':show_lines,'Power_curve':make_power_curve(num_raters)}

    return  items


def make_classifier_scores(colors,base_names,num_of_classifiers):

    confidence_radius = np.array(np.sort(random.sample(list(np.random.uniform(0.005, 0.01,10)),num_of_classifiers))).reshape(num_of_classifiers,1)
    scores = np.array(np.sort(random.sample(list(np.random.uniform(0.05, 0.99,15)),num_of_classifiers))).reshape(num_of_classifiers,1)
    colors = np.array(colors).reshape(num_of_classifiers,1)
    base_names = np.array(base_names).reshape(num_of_classifiers,1)
    data = (np.hstack((base_names,colors,scores,confidence_radius)))


    dataframe=pd.DataFrame(data, columns=['names','colors','scores','confidence_radius'])


    return  dataframe



def make_points_to_show_surveyEquiv(x_value,am=None,classifier=None):
    surveyEquivlst =[]
    surveyEquiv={}
    if classifier is not None:
        surveyEquiv['which_type'] = 'classifier'
        surveyEquiv['which_x_value']=x_value
        surveyEquiv['name'] = classifier
        surveyEquivlst.append(surveyEquiv)
    if am is not None:
        surveyEquiv['which_type'] = 'amateur'
        surveyEquiv['which_x_value']=x_value
        surveyEquiv['name'] =am
        surveyEquivlst.append(surveyEquiv)
    return surveyEquivlst
