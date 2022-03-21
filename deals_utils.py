import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def max_substrategy_match(a,b):
    if a == b:
        return True
    else:
        return False

def max_substrategy(df):
    df_nov_exp_6_substrategy_total = pd.pivot_table(df, index = ['PORTFOLIONR'],columns=['SUBSTRATEGY'], values = ['BOOK_VALUE_REF'], 
                                       aggfunc= {np.sum})
    df_nov_exp_6_substrategy_total.fillna(0, inplace= True)
    df_nov_exp_6_substrategy_total.reset_index(inplace=True, drop = False)

    cols_list_substrategy = []
    for i in df_nov_exp_6_substrategy_total.columns:
        if i[1] == '':
            cols_list_substrategy.append(i[0])
        else:
    #         cols_list_substrategy.append(i[1] + "_" + i[0])
            cols_list_substrategy.append(i[2])

    df_nov_exp_6_substrategy_total.columns = cols_list_substrategy

    df_nov_exp_6_substrategy_total['Max_SubStrategy'] = df_nov_exp_6_substrategy_total[['CG', 'CY']].idxmax(axis=1)
    df_nov_exp_6_substrategy_total['Max_SubStrategy_Amount'] = df_nov_exp_6_substrategy_total[['CG', 'CY']].max(axis=1)

    df_mapping_Portfolio_Max_Substrategy = df_nov_exp_6_substrategy_total[['PORTFOLIONR','Max_SubStrategy']]
    df_return = pd.merge(df, df_mapping_Portfolio_Max_Substrategy, on = ['PORTFOLIONR'], how = 'left')
    df_return = df_return[[ 'PORTFOLIONR', 'ISIN','MASTERISIN','SUBSTRATEGY', 
       'MARKET_VALUE_REF',  'BOOK_VALUE_REF','Date_Diff', 'Less_Than_2_Yrs', 'Max_SubStrategy']]
    df_return['Is_MaxSubstrategy'] = df_return.apply(lambda x:max_substrategy_match(x['SUBSTRATEGY'], 
                                                                                                     x['Max_SubStrategy']), axis=1)
    
    df_return_justmax_strategy = df_return.copy()
    df_return_justmax_strategy.reset_index(inplace= True, drop = True)
    df_return_justmax_strategy['Recent_Phase'] = np.where(df_return_justmax_strategy['Date_Diff']<180, True, False)
    df_return_justmax_strategy = df_return_justmax_strategy[df_return_justmax_strategy['Is_MaxSubstrategy'] == True]
    df_return_justmax_strategy.reset_index(inplace = True, drop = True)
    
    return(df_return_justmax_strategy)

def return_min(df, cols, buffer_liquidity):
    min_value = min(df[cols]) 
    adjusted_min = min_value - ((buffer_liquidity*(df['AUM']))/100)
    if adjusted_min > 0: 
        return adjusted_min
    else:
        return 0

def deal_sizing(clientid, expo_df, max_substrat_df, deals_df, client_deals, topup, top_n_deals,top_aum_criteria_perc, min_value_topup):

    # try:
    deals_client_sub = list(client_deals[clientid])

    if topup == False:
        deals_available_sub = deals_df[~deals_df['MASTERISIN'].isin(deals_client_sub)]
    elif topup == True:
        deals_available_sub = deals_df.copy()

    max_substrat_df.reset_index(inplace = True, drop = True)

    max_substrategy = max_substrat_df['Max_SubStrategy'][0]
    print(max_substrategy)

    expo_df = pd.pivot_table(expo_df, index = ['PORTFOLIONR','MASTERISIN'], values = ['MARKET_VALUE_REF'], aggfunc = np.sum)
    expo_df.columns = ['MARKET_VALUE_REF']
    expo_df.reset_index(inplace = True, drop = False)
    print(expo_df)
    
    if max_substrategy == 'CG':
        deals_available_sub_strategy = deals_available_sub[deals_available_sub['Capital Growth'] == 'G']
        # print(deals_available_sub)
    elif max_substrategy == 'CY':
        deals_available_sub_strategy = deals_available_sub[deals_available_sub['Capital Yielding'] == 'Y']
        # print(deals_available_sub)
    deals_available_sub_strategy.reset_index(inplace= True, drop = True)
    deals_available_sub_strategy.sort_values(by = ['Less_Than_6_Months', 'Expected MOIC'], ascending= [False, False], inplace = True)

    deals_available_sub_strategy['PORTFOLIONR'] = max_substrat_df['PORTFOLIONR'][0]

    output_df = deals_available_sub_strategy[['PORTFOLIONR', 'MASTERISIN', 'Dry Powder/ Available amount' ]]

    drypowder_deal = output_df['Dry Powder/ Available amount'][0]
    

    # print(expo_df.columns)
    deals_available_sub_strategy_upd = pd.merge(output_df, expo_df[['PORTFOLIONR', 'MASTERISIN', 'MARKET_VALUE_REF']], on = ['PORTFOLIONR', 'MASTERISIN'], how = 'left')
    deals_available_sub_strategy_upd = pd.merge(deals_available_sub_strategy_upd, max_substrat_df, on = ['PORTFOLIONR'], how = 'left')
    # deals_available_sub_strategy_upd = pd.merge(output_df, max_substrat_df, on = ['PORTFOLIONR'], how = 'left')

    aum_value = deals_available_sub_strategy_upd['AUM'][0]  
    initial_liquidity = deals_available_sub_strategy_upd['Min_Liquidity'][0]
    print('1 Check')
    print(len(deals_available_sub_strategy_upd))
    deals_available_sub_strategy_upd.reset_index(inplace= True, drop = True)
    deals_available_sub_strategy_upd.fillna(0, inplace= True)

    median = deals_available_sub_strategy_upd['Median_BOOK_VALUE_REF'][0]
    deals_available_sub_strategy_upd['DEAL_DRYPOWDER_MORE_ALLOCATION'] = 0
    deals_available_sub_strategy_upd['DEAL_DRYPOWDER_MORE_ALLOCATION'] = deals_available_sub_strategy_upd['Dry Powder/ Available amount'] - median
    deals_available_sub_strategy_upd['DEAL_DRYPOWDER_MORE_ALLOCATION_Boolean'] = True
    deals_available_sub_strategy_upd['DEAL_DRYPOWDER_MORE_ALLOCATION_Boolean'] = np.where(deals_available_sub_strategy_upd['DEAL_DRYPOWDER_MORE_ALLOCATION'] >0, True, False)
    print('2 Check')
    print(len(deals_available_sub_strategy_upd))
    print(deals_available_sub_strategy_upd)
    deals_available_sub_strategy_upd = deals_available_sub_strategy_upd[deals_available_sub_strategy_upd['DEAL_DRYPOWDER_MORE_ALLOCATION_Boolean'] == True]
    deals_available_sub_strategy_upd.reset_index(inplace = True, drop = True)
    print('3 Check')
    print(len(deals_available_sub_strategy_upd))
    print(deals_available_sub_strategy_upd.columns)
    print('deals upd')
    print(deals_available_sub_strategy_upd)
    deals_available_sub_strategy_upd['Delta_allocation'] = deals_available_sub_strategy_upd['Median_BOOK_VALUE_REF'] - deals_available_sub_strategy_upd['MARKET_VALUE_REF']
    print(len(deals_available_sub_strategy_upd))
    print(deals_available_sub_strategy_upd)
    deals_available_sub_strategy_upd= deals_available_sub_strategy_upd[deals_available_sub_strategy_upd['Delta_allocation'] > 0]
    # deals_available_sub_strategy_upd.reset_index(inplace = True, drop = True)
    print(len(deals_available_sub_strategy_upd))
    
    # deals_available_sub_strategy_upd['TOPUP_Active'] = np.where(np.logical_and(deals_available_sub_strategy_upd['MARKET_VALUE_REF'] > 0 , np.logical_and(deals_available_sub_strategy_upd['Delta_allocation'] < 0.001* aum_value, deals_available_sub_strategy_upd['Delta_allocation'] < 100000)), False, True)
    deals_available_sub_strategy_upd['TOPUP_Active'] = np.where(np.logical_and(deals_available_sub_strategy_upd['MARKET_VALUE_REF'] > 0 , np.logical_and(deals_available_sub_strategy_upd['Delta_allocation'] < ((top_aum_criteria_perc* aum_value)/100), deals_available_sub_strategy_upd['Delta_allocation'] < min_value_topup)), False, True)


    deals_available_sub_strategy_upd = deals_available_sub_strategy_upd[deals_available_sub_strategy_upd['TOPUP_Active'] == True]
    deals_available_sub_strategy_upd.reset_index(inplace = True, drop = True)

    deals_available_sub_strategy_upd['UPD_MIN_Liquidity'] = initial_liquidity
    deals_available_sub_strategy_upd['UPD_Liquidity'] = 0
    deals_available_sub_strategy_upd['FilterLiquidity'] = True

    for i in range(len(deals_available_sub_strategy_upd)):
        if i == 0:
            deals_available_sub_strategy_upd['UPD_Liquidity'][i] = initial_liquidity - deals_available_sub_strategy_upd['Delta_allocation'][i]
            deals_available_sub_strategy_upd['FilterLiquidity'][i] = True
        else:
            deals_available_sub_strategy_upd['UPD_Liquidity'][i] = deals_available_sub_strategy_upd['UPD_Liquidity'][i-1] - deals_available_sub_strategy_upd['Delta_allocation'][i]
            if deals_available_sub_strategy_upd['UPD_Liquidity'][i-1] < 0: 
                deals_available_sub_strategy_upd['FilterLiquidity'][i] = False
            else: 
                deals_available_sub_strategy_upd['FilterLiquidity'][i] = True

    deals_available_sub_strategy_upd = deals_available_sub_strategy_upd[deals_available_sub_strategy_upd['FilterLiquidity'] == True]
    deals_available_sub_strategy_upd.reset_index(inplace = True, drop = True)
    deals_available_sub_strategy_upd['VALUE'] = 1
    deals_available_sub_strategy_upd['Recommendation'] = deals_available_sub_strategy_upd.groupby(['PORTFOLIONR'])['VALUE'].cumsum()
    return deals_available_sub_strategy_upd
    # except:
    #     pass    

def current_allocation_underweight(df, df_client_ips):
    df_nov_exp_4_allocation = pd.pivot_table(df, index = ['PORTFOLIONR'], columns = ['SUBSTRATEGY'], values = ['MARKET_VALUE_REF'], aggfunc = np.sum)

    list_cols = []
    for i in range(len(df_nov_exp_4_allocation.columns)):
        cols = df_nov_exp_4_allocation.columns[i][0] + '_' + df_nov_exp_4_allocation.columns[i][1]
        list_cols.append(cols)

    df_nov_exp_4_allocation.columns = list_cols
    df_nov_exp_4_allocation.reset_index(inplace= True, drop = False)
    df_nov_exp_4_allocation.fillna(0, inplace= True)

    df_nov_exp_4_allocation['Total_allocation'] = df_nov_exp_4_allocation['MARKET_VALUE_REF_AR'] + df_nov_exp_4_allocation['MARKET_VALUE_REF_CG'] + df_nov_exp_4_allocation['MARKET_VALUE_REF_CY'] + df_nov_exp_4_allocation['MARKET_VALUE_REF_OP']

    df_nov_exp_4_allocation['Current_AR%'] = np.array(df_nov_exp_4_allocation['MARKET_VALUE_REF_AR'])/ np.array(df_nov_exp_4_allocation['Total_allocation'])
    df_nov_exp_4_allocation['Current_CG%'] = np.array(df_nov_exp_4_allocation['MARKET_VALUE_REF_CG'])/ np.array(df_nov_exp_4_allocation['Total_allocation'])
    df_nov_exp_4_allocation['Current_CY%'] = np.array(df_nov_exp_4_allocation['MARKET_VALUE_REF_CY'])/ np.array(df_nov_exp_4_allocation['Total_allocation'])
    df_nov_exp_4_allocation['Current_OP%'] = np.array(df_nov_exp_4_allocation['MARKET_VALUE_REF_OP'])/ np.array(df_nov_exp_4_allocation['Total_allocation'])

    df_nov_exp_4_allocation_ips = pd.merge(df_nov_exp_4_allocation, df_client_ips, left_on= ['PORTFOLIONR'], right_on= ['Mandate'], how = 'left')

    df_nov_exp_4_allocation_ips['Delta_AR'] = df_nov_exp_4_allocation_ips['AR%'] - df_nov_exp_4_allocation_ips['Current_AR%']
    df_nov_exp_4_allocation_ips['Delta_CG'] = df_nov_exp_4_allocation_ips['CG%'] - df_nov_exp_4_allocation_ips['Current_CG%']
    df_nov_exp_4_allocation_ips['Delta_CY'] = df_nov_exp_4_allocation_ips['CY%'] - df_nov_exp_4_allocation_ips['Current_CY%']
    df_nov_exp_4_allocation_ips['Delta_OP'] = df_nov_exp_4_allocation_ips['Opp%'] - df_nov_exp_4_allocation_ips['Current_OP%']

    df_nov_exp_4_allocation_ips['Suggest_AR'] = np.where(df_nov_exp_4_allocation_ips['Delta_AR']>0, True, False)
    df_nov_exp_4_allocation_ips['Suggest_CG'] = np.where(df_nov_exp_4_allocation_ips['Delta_CG']>0, True, False)
    df_nov_exp_4_allocation_ips['Suggest_CY'] = np.where(df_nov_exp_4_allocation_ips['Delta_CY']>0, True, False)
    df_nov_exp_4_allocation_ips['Suggest_OP'] = np.where(df_nov_exp_4_allocation_ips['Delta_OP']>0, True, False)

    return df_nov_exp_4_allocation_ips

def deal_recommendation_ips(expo_df, suggest_substrat_df, top_n_deals, deals_available_sub_strategy, substrategy_def, Updated_deal_allocation_aum, top_aum_criteria_perc,  min_value_topup):
    deals_available_sub_strategy.reset_index(inplace= True, drop = True)
    print(deals_available_sub_strategy)
    deals_available_sub_strategy.sort_values(by = ['Less_Than_6_Months', 'Expected MOIC'], ascending= [False, False], inplace = True)
    deals_available_sub_strategy['PORTFOLIONR'] = suggest_substrat_df['PORTFOLIONR'][0]
    output_df = deals_available_sub_strategy[['PORTFOLIONR', 'MASTERISIN', 'Dry Powder/ Available amount' ]]
    drypowder_deal = output_df['Dry Powder/ Available amount'][0]
    print(drypowder_deal)
    
    deals_available_sub_strategy_upd = pd.merge(output_df, expo_df[['PORTFOLIONR', 'MASTERISIN', 'MARKET_VALUE_REF']], on = ['PORTFOLIONR', 'MASTERISIN'], how = 'left')
    
    print('check 0.1')
    print(deals_available_sub_strategy_upd)
    
    deals_available_sub_strategy_upd = pd.merge(deals_available_sub_strategy_upd, suggest_substrat_df, on = ['PORTFOLIONR'], how = 'left')
    
    print('check 0.2')
    
    print(deals_available_sub_strategy_upd)
    deals_available_sub_strategy_upd.drop_duplicates(inplace = True)
    deals_available_sub_strategy_upd.reset_index(inplace= True, drop = True)
    deals_available_sub_strategy_upd.fillna(0, inplace= True)
    
    print('check 0')
    print(deals_available_sub_strategy_upd)
    
    median_col = 'Median_Book_' + substrategy_def
    
    aum_value = deals_available_sub_strategy_upd['AUM'][0]
    median = deals_available_sub_strategy_upd[median_col][0]
    min_liquidity = deals_available_sub_strategy_upd['Min_Liquidity'][0]
    if median == 0:
        median = (Updated_deal_allocation_aum*aum_value)/100
        
    deals_available_sub_strategy_upd['UPD_MEDIAN'] = median
    
    deals_available_sub_strategy_upd['DEAL_DRYPOWDER_MORE_ALLOCATION'] = 0
    deals_available_sub_strategy_upd['DEAL_DRYPOWDER_MORE_ALLOCATION'] = deals_available_sub_strategy_upd['Dry Powder/ Available amount'] - median
    deals_available_sub_strategy_upd['DEAL_DRYPOWDER_MORE_ALLOCATION_Boolean'] = True
    deals_available_sub_strategy_upd['DEAL_DRYPOWDER_MORE_ALLOCATION_Boolean'] = np.where(deals_available_sub_strategy_upd['DEAL_DRYPOWDER_MORE_ALLOCATION'] >0, True, False)
    print('deals_available_sub_strategy_upd')
    print(deals_available_sub_strategy_upd)
    deals_available_sub_strategy_upd = deals_available_sub_strategy_upd[deals_available_sub_strategy_upd['DEAL_DRYPOWDER_MORE_ALLOCATION_Boolean'] == True]
    deals_available_sub_strategy_upd.reset_index(inplace = True, drop = True)
    print('check_len 1')
    print(len(deals_available_sub_strategy_upd))
    print(deals_available_sub_strategy_upd)
    deals_available_sub_strategy_upd['Delta_allocation'] = (deals_available_sub_strategy_upd['MARKET_VALUE_REF'] - median)*(-1)
    print(deals_available_sub_strategy_upd)
    print('Delta Allocation')
    deals_available_sub_strategy_upd= deals_available_sub_strategy_upd[deals_available_sub_strategy_upd['Delta_allocation'] > 0]
    
    deals_available_sub_strategy_upd.reset_index(inplace =True, drop = True)
    print('aum_value')
    print(aum_value*0.001)
    print('check_len 2')
    print(len(deals_available_sub_strategy_upd))
    print(deals_available_sub_strategy_upd)
    initial_liquidity = min_liquidity
    
    deals_available_sub_strategy_upd['TOPUP_Active'] = np.where(np.logical_and(deals_available_sub_strategy_upd['MARKET_VALUE_REF'] > 0 , np.logical_and(deals_available_sub_strategy_upd['Delta_allocation'] < ((top_aum_criteria_perc* aum_value)/100), deals_available_sub_strategy_upd['Delta_allocation'] < min_value_topup)), False, True)
    print('check_len 3')
    print(deals_available_sub_strategy_upd)
    deals_available_sub_strategy_upd = deals_available_sub_strategy_upd[deals_available_sub_strategy_upd['TOPUP_Active'] == True]
    deals_available_sub_strategy_upd.reset_index(inplace = True, drop = True)
    
    print('check_len 4')
    print(len(deals_available_sub_strategy_upd))
    
    deals_available_sub_strategy_upd['UPD_MIN_Liquidity'] = initial_liquidity
    deals_available_sub_strategy_upd['UPD_Liquidity'] = 0
    deals_available_sub_strategy_upd['FilterLiquidity'] = True
    
    for i in range(len(deals_available_sub_strategy_upd)):
        if i == 0:
            deals_available_sub_strategy_upd['UPD_Liquidity'][i] = initial_liquidity - deals_available_sub_strategy_upd['Delta_allocation'][i]
            deals_available_sub_strategy_upd['FilterLiquidity'][i] = True
        else:
            deals_available_sub_strategy_upd['UPD_Liquidity'][i] = deals_available_sub_strategy_upd['UPD_Liquidity'][i-1] - deals_available_sub_strategy_upd['Delta_allocation'][i]
            if deals_available_sub_strategy_upd['UPD_Liquidity'][i-1] < 0: 
                deals_available_sub_strategy_upd['FilterLiquidity'][i] = False
            else: 
                deals_available_sub_strategy_upd['FilterLiquidity'][i] = True
        
    deals_available_sub_strategy_upd = deals_available_sub_strategy_upd[deals_available_sub_strategy_upd['FilterLiquidity'] == True]
    deals_available_sub_strategy_upd.reset_index(inplace = True, drop = True)
    deals_available_sub_strategy_upd['VALUE'] = 1
    deals_available_sub_strategy_upd['SUBSTRATEGY'] = substrategy_def
    deals_available_sub_strategy_upd['Recommendation'] = deals_available_sub_strategy_upd.groupby(['PORTFOLIONR'])['VALUE'].cumsum()
    return deals_available_sub_strategy_upd

def deal_sizing_ips(clientid,expo_df_def1, suggest_substrat_df_def1, deals_df_def1, client_deals_def1, topup, top_n_deals_def1,  Updated_deal_allocation_aum, top_aum_criteria_perc, min_value_topup):
    
    deals_client_sub = list(client_deals_def1[clientid])

    if topup == False:
        deals_available_sub = deals_df_def1[~deals_df_def1['MASTERISIN'].isin(deals_client_sub)]
    elif topup == True:
        deals_available_sub = deals_df_def1.copy()

    suggest_substrat_df_def1.reset_index(inplace = True, drop = True)

    expo_df_def1 = pd.pivot_table(expo_df_def1, index = ['PORTFOLIONR','MASTERISIN'], values = ['MARKET_VALUE_REF'], aggfunc = np.sum)
    expo_df_def1.columns = ['MARKET_VALUE_REF']
    expo_df_def1.reset_index(inplace = True, drop = False)

    print(suggest_substrat_df_def1['Suggest_CG'][0])

    if suggest_substrat_df_def1['Suggest_CG'][0] == True:
        deals_available_sub_strategy_def1 = deals_available_sub[deals_available_sub['Capital Growth'] == 'G']
        print(len(deals_available_sub_strategy_def1))
        final_cg_df = deal_recommendation_ips(expo_df = expo_df_def1, suggest_substrat_df = suggest_substrat_df_def1, top_n_deals = top_n_deals_def1, deals_available_sub_strategy = deals_available_sub_strategy_def1, substrategy_def = 'CG', Updated_deal_allocation_aum= Updated_deal_allocation_aum, top_aum_criteria_perc = top_aum_criteria_perc, min_value_topup = min_value_topup)

    if suggest_substrat_df_def1['Suggest_CY'][0] == True:
        deals_available_sub_strategy_def1 = deals_available_sub[deals_available_sub['Capital Yielding'] == 'Y']   
        final_cy_df = deal_recommendation_ips(expo_df = expo_df_def1, suggest_substrat_df = suggest_substrat_df_def1, top_n_deals = top_n_deals_def1, deals_available_sub_strategy = deals_available_sub_strategy_def1, substrategy_def = 'CY', Updated_deal_allocation_aum= Updated_deal_allocation_aum, top_aum_criteria_perc = top_aum_criteria_perc, min_value_topup = min_value_topup)

    if np.logical_and(suggest_substrat_df_def1['Suggest_CG'][0] == True, suggest_substrat_df_def1['Suggest_CY'][0] == True):
        ips_df = pd.concat([final_cg_df, final_cy_df])
        return(ips_df)
    elif np.logical_and(suggest_substrat_df_def1['Suggest_CG'][0] == False, suggest_substrat_df_def1['Suggest_CY'][0] == True):
        ips_df = final_cy_df.copy()
        return(ips_df)
    elif np.logical_and(suggest_substrat_df_def1['Suggest_CG'][0] == True, suggest_substrat_df_def1['Suggest_CY'][0] == False):
        ips_df = final_cg_df.copy()
        return(ips_df)

def filter_last_2years(df):
    df['Date_Today'] = datetime.today()
    # datetime.today().strftime("%Y-%m-%d")
    df['Date_Diff'] = df['Date_Today']- df['Date']
    df['Date_Diff'] = df['Date_Diff']/ np.timedelta64(1,'D')
    df['Less_Than_2_Yrs'] = np.where(df['Date_Diff']<2*365, True, False)
    df_return = df[df['Less_Than_2_Yrs'] == True]

    return df_return
