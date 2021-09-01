# Any needed imports
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind, ttest_ind_from_stats
import numpy as np
from scipy import stats
import seaborn as sns
from scipy.special import logit, expit
import scipy

class PlayGraphs:
    """
    This class contains methods for generating histograms, errorbars, and t-tests for a play.     
    """

    def __init__(self, play_df, figure_title=None):
      """
      play_df should be a dataframe with WNLU data and should have the 
      'gender' and/or 'Outsider_Present' and/or 'Nobility_Present' column(s).

      figure_title - optional string used for naming the saved figures/t-tests
      """
      self.play_df = play_df
      if figure_title is not None:
        self.figure_title = figure_title
      else:
        self.figure_title = 'example'

      self.ttests = []

    def preprocess_data(self):
      """
      Processes dataframe to have correct column names and data types
      """
      play_df = self.play_df
      # combining any columns that should have same name
      score_types = ['anger','disgust','fear','joy','sadness','sentiment']
      score_types2 = ['emotion_anger','emotion_disgust','emotion_fear','emotion_joy','emotion_sadness','sentiment_score']
      for score_type, score_type2 in zip(score_types, score_types2):
        try:
          play_df[score_type] = play_df[score_type].combine_first(play_df[score_type2])
        except:
          play_df[score_type] = play_df[score_type2]


      # changes gender col to none if speaker is none
      try:
        play_df['gender'] = play_df.apply(lambda x: x['speaker'] if pd.isnull(x['speaker']) else x['gender'], axis=1)
      except:
        pass

      # Drop rows with no WNLU data
      play_df = play_df[play_df['nlu_status']=='pass']


      self.play_df = play_df


    def generate_histograms(self, label_type = None):
      """
      label_type - Should be 'gender', 'outsider', or 'class'
      This function generates histograms for each WNLU emotion score + sentiment.

      Does this analysis for one label_type that is specified:
        gender - play_df must have gender column (compares male v female)
        outsider - play_df must have Outsider_Present column (outsider v insider)
        class - play_df must have Nobility_Present column (nobility v commoners)
      """
      
      play_df = self.play_df
      fig = plt.figure(figsize=(17,24))

      # splitting lines into two groups
      if label_type == 'gender':
        lines_first_type = play_df[play_df['gender'] =='male']
        lines_second_type = play_df[play_df['gender'] =='female']
      elif label_type == 'outsider':
        lines_first_type = play_df[play_df['Outsider_Present'] == True]
        lines_second_type = play_df[play_df['Outsider_Present'] ==False]
      else:
        lines_first_type = play_df[play_df['Nobility_Present'] == True]
        lines_second_type = play_df[play_df['Nobility_Present'] ==False]


      score_types = ['anger','disgust','fear','joy','sadness','sentiment']
      
  
      for score_type,num in zip(score_types, range(1,7)):
        

        logitname = 'logit_'+str(score_type)

        # converts data to logit distribution for histograms
        lines_first_type[logitname] = logit(lines_first_type[score_type])
        lines_second_type[logitname] = logit(lines_second_type[score_type])

        logit_df_1 = lines_first_type.replace([np.inf, -np.inf], np.nan)
        logit_df_2 = lines_second_type.replace([np.inf, -np.inf], np.nan)


        ax = fig.add_subplot(6,2,num)
        logit_score = 'logit_'+str(score_type)
        
        # plots for different label_types
        if label_type == 'gender':
          sns.distplot(logit_df_1[logit_score], fit=scipy.stats.norm, kde=False, hist=True,fit_kws={"color":"red"}, color='r',ax=ax,label='male').set_title('Logit of '+str(score_type.capitalize()))
          sns.distplot(logit_df_2[logit_score], fit=scipy.stats.norm, kde=False, hist=True,fit_kws={"color":"blue"}, color='b',ax=ax,label='female')
        elif label_type == 'outsider':
          sns.distplot(logit_df_1[logit_score], fit=scipy.stats.norm, kde=False, hist=True,fit_kws={"color":"green"}, color='g',ax=ax,label='Outsider').set_title('Logit of '+str(score_type.capitalize()))
          sns.distplot(logit_df_2[logit_score], fit=scipy.stats.norm, kde=False, hist=True,fit_kws={"color":"purple"}, color='purple',ax=ax,label='Insider')
        else:
          sns.distplot(logit_df_1[logit_score], fit=scipy.stats.norm, kde=False, hist=True,fit_kws={"color":"gold"}, color='gold',ax=ax,label='Nobility').set_title('Logit of '+str(score_type.capitalize()))
          sns.distplot(logit_df_2[logit_score], fit=scipy.stats.norm, kde=False, hist=True,fit_kws={"color":"firebrick"}, color='firebrick',ax=ax,label='Commoner')
        ax.legend()
        
      plt.subplots_adjust(hspace=.45)

      fig_title = self.figure_title + '_' + str(label_type) + '_histograms.png'

      fig.savefig(fig_title, bbox_inches='tight')

    def generate_errorbars(self, label_type):
      """
      label_type - Should be 'gender', 'outsider', or 'class'
      This function generates errorbars and two-sided t-tests for each WNLU emotion score + sentiment.

      Does this analysis for one label_type that is specified:
        gender - play_df must have gender column (compares male v female)
        outsider - play_df must have Outsider_Present column (outsider v insider)
        class - play_df must have Nobility_Present column (nobility v commoners)
      """

      play_df = self.play_df

      fig = plt.figure(figsize=(17,24))
      score_types = ['anger','disgust','fear','joy','sadness','sentiment']
      self.ttests = []
      
      # splitting data into two groups
      if label_type == 'gender':
        lines_first_type = play_df[play_df['gender'] =='male']
        lines_second_type = play_df[play_df['gender'] =='female']
      elif label_type == 'outsider':
        lines_first_type = play_df[play_df['Outsider_Present'] == True]
        lines_second_type = play_df[play_df['Outsider_Present'] ==False]
      else:
        lines_first_type = play_df[play_df['Nobility_Present'] == True]
        lines_second_type = play_df[play_df['Nobility_Present'] == False]

      for score_type,num in zip(score_types, range(1,7)):
        ttest = {}
        logitname = 'logit_'+str(score_type)
        
        lines_first_type[logitname] = logit(lines_first_type[score_type])
        lines_second_type[logitname] = logit(lines_second_type[score_type])

        logit_df_1 = lines_first_type.replace([np.inf, -np.inf], np.nan)
        logit_df_2 = lines_second_type.replace([np.inf, -np.inf], np.nan)

        ax = fig.add_subplot(6,2,num)
        logit_score = 'logit_'+str(score_type)

        # calculating standard error
        stderror_m=np.std(logit_df_1[logit_score])/np.sqrt(len(logit_df_1[logit_score]))
        stderror_f=np.std(logit_df_2[logit_score])/np.sqrt(len(logit_df_2[logit_score]))

        if label_type == 'outsider':
          plt.errorbar(np.mean(logit_df_1[logit_score]),0.05,xerr=3*stderror_m,fmt='|', ms=30,mew=2,capthick=2,capsize=10,color='green',label='Outsider')
          plt.title(str(logit_score).capitalize())

          plt.errorbar(np.mean(logit_df_2[logit_score]),0.1,xerr=3*stderror_f,fmt='|', ms=30,mew=2,capthick=2,capsize=10,color='purple',label='Insider')
        
        elif label_type == 'gender':
          plt.errorbar(np.mean(logit_df_1[logit_score]),0.05,xerr=3*stderror_m,fmt='|', ms=30,mew=2,capthick=2,capsize=10,color='red',label='male')
          plt.title(str(logit_score).capitalize())
          plt.errorbar(np.mean(logit_df_2[logit_score]),0.1,xerr=3*stderror_f,fmt='|', ms=30,mew=2,capthick=2,capsize=10,color='blue',label='female')
        
        else:
          plt.errorbar(np.mean(logit_df_1[logit_score]),0.05,xerr=3*stderror_m,fmt='|', ms=30,mew=2,capthick=2,capsize=10,color='gold',label='Nobility')
          plt.title(str(logit_score).capitalize())
          plt.errorbar(np.mean(logit_df_2[logit_score]),0.1,xerr=3*stderror_f,fmt='|', ms=30,mew=2,capthick=2,capsize=10,color='firebrick',label='Commoners')
        
        ax.legend()
        plt.ylim([-.02,.15])
        
        t, p = ttest_ind(logit_df_1[score_type].dropna(), logit_df_2[score_type].dropna(), equal_var=False)


        ttest['score_type'] = score_type
        ttest['t'] = t
        ttest['p'] = p
        self.ttests.append(ttest)
        left, right = plt.xlim()
        p = np.format_float_positional(p, precision=3, unique=False, fractional=False, trim='k')


        p_score_str = 'p value: ' + str(p)
        plt.text(left+.006,0,p_score_str,style='italic', fontsize=12, bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 5})

      fig_title = self.figure_title + '_' + str(label_type) + '_errorbars.png'

      fig.savefig(fig_title, bbox_inches='tight')
      
      ttest_data = pd.DataFrame(self.ttests)
      display(ttest_data)

      ttest_title = self.figure_title + '_' + str(label_type) + '_ttests.csv'

      ttest_data.to_csv(ttest_title)

