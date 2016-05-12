import os
import pandas as pd
import numpy as np
import zipfile
import argparse
import datetime
from datetime import date, timedelta
from sqlalchemy import create_engine
from pydblite import Base


# DF with predicted categories
def get_predicted_data(path_to_file, is_zip=False):
    if is_zip:
        file_name = '.'.join(path_to_file.split('/')[-1].split('.')[:2])
        with zipfile.ZipFile(path_to_file, 'r') as z:
            f = z.open(file_name)
            return pd.read_csv(f)
    else:
        with open(path_to_file, 'r') as f:
            return pd.read_csv(f)


# Strip non ascii chars
def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)
        
# Update(create) pylitedb from A_Master
def update_database(db, connection_string):
    print 'Start to update'
    now = datetime.datetime.now().date()
    offset = (now.weekday() - 5) % 7
    last_saturday = now - timedelta(days=offset)
    if db:
        indexes = [record.get('__id__') for record in db]
        last_index = max(indexes)
        update_from = db[last_index]['key'].sort_values('Date', ascending=False).reset_index(drop=True)['Date'].get(0)
    else:
        update_from = date(year=2014, month=1, day=1)
    if last_saturday == update_from:
        print 'Base is up to date.'
    else:
        if db:
            update_from = update_from + timedelta(days=1)
        print 'Pylitedb will updated from {} to {}'.format(update_from, last_saturday)
    engine = create_engine(connection_string)
    row_sql = 'select SKUConfig, date_sub(Date,INTERVAL WEEKDAY(Date) -5 DAY) Date, PaidPrice from A_Master where Date >= "{}" and Date <= "{}"'.format(update_from, last_saturday)
    df = pd.read_sql(row_sql, engine)
    print 'We took DataFrame from master database, try to insert in pylitedb.'
    if df.empty:
        print 'DataFrame is empty. No data from master database. Can\'t update pylitedb.'
        return False
    db.insert(df)
    db.commit()
    print 'Done'
    return df
        

# Sales data
def load_sales_data(db, cat, connection_string='', path_to_file='', is_sql=False, row_sql=''):
    if is_sql:
        engine = create_engine(connection_string)
        df = pd.read_sql(row_sql,engine)
    else:
        df = pd.DataFrame()
        for record in db:
            df = df.append(record['key'])
    print 'Data loaded'
    return df.rename(columns={''.format(cat): 'Cat'})


# Joined sales data with predicted categories
def join_sales_predict(sales_data, predicted_data):
    merged_data = sales_data.merge(predicted_data, left_on='SKUConfig', right_on='sku')
    return merged_data.ix[:,['pred_cat_name','PaidPrice','Date','sku']].groupby(['pred_cat_name','Date']).agg({'sku':'count','PaidPrice':'sum'})


# Prepared data with sales
def prepare(data):
    data['Cat'] = [i[0] for i in data.index]
    data['Date'] = [i[1] for i in data.index]
    data.index = range(len(data))
    data.columns = ['items_sold','PaidPrice','Cat','Date']
    data['Cat'] = data.Cat.apply(lambda x: strip_non_ascii(x))
    return data


# Top categories by paid price
def categorization(data, file_name):
    categorized_data = data.groupby('Cat').sum().sort_values('PaidPrice',ascending = False)
    categorized_data['Cat'] = [i for i in categorized_data.index]
    categorized_data.index = range(len(categorized_data))
    categorized_data.to_csv(file_name)
    print '{} created'.format(file_name)
    return categorized_data
    

# Save top_n cat to csv file
def save_top_cat(cat_data, prep_data, top_n, file_name):
    top_cats = prep_data.ix[prep_data.Cat.apply(lambda x: x in set(cat_data.ix[range(top_n), 'Cat'])),:]
    top_cats.ix[:,['Cat','Date','PaidPrice','items_sold']].to_csv(file_name)
    print '{} created'.format(file_name)
    return top_cats


# Functions to prepare dataframe with forecast
def prepare_forecast_items(df,df_items, category):
    df1 = df.ix[df.Cat == category,:]
    items = np.array(df1.items_sold)
    
    df1_f = df_items.ix[df_items.Cat == category,:]
    if len(df1_f) > 0:
        items_pess = np.array(df1_f['pessimistic'])
        items_avg = np.array(df1_f['average'])
        items_opt = np.array(df1_f['optimistic'])
    else:
        items_pess = np.array(['NA','NA'])
        items_avg = np.array(['NA','NA'])
        items_opt = np.array(['NA','NA'])
    
    df2 = pd.DataFrame({'last 24 weeks':[0],'last 12 weeks':[0],'last 4 weeks':[0],'last week':[0],\
                        'next week pessimistic':[0],'next week average':[0],'next week optimistic':[0],\
                        '2 weeks ahead pessimistic':[0],'2 weeks ahead average':[0],'2 weeks ahead optimistic':[0]})
    df2.index = ['{}'.format(category)]
    df2['last 24 weeks'] = round(np.mean(items[-24:]))
    df2['last 12 weeks'] = round(np.mean(items[-12:]))
    df2['last 4 weeks'] = round(np.mean(items[-4:]))
    df2['last week'] = round(np.mean(items[-1:]))
    df2['next week pessimistic'] = items_pess[0]
    df2['next week average'] = items_avg[0]
    df2['next week optimistic'] = items_opt[0]
    df2['2 weeks ahead pessimistic'] = items_pess[1]
    df2['2 weeks ahead average'] = items_avg[1]
    df2['2 weeks ahead optimistic'] = items_opt[1]
    return df2


def prepare_forecast_price(df,df_price, category):
    df1 = df.ix[df.Cat == category,:]
    price = np.array(df1.PaidPrice)
    
    # forecast = call to R function
    df1_f = df_price.ix[df_price.Cat == category,:]
    if len(df1_f) > 0:
        price_pess = np.array(df1_f['pessimistic'])
        price_avg = np.array(df1_f['average'])
        price_opt = np.array(df1_f['optimistic'])
    else:
        price_pess = np.array(['NA','NA'])
        price_avg = np.array(['NA','NA'])
        price_opt = np.array(['NA','NA'])
    
    df2 = pd.DataFrame({'last 24 weeks':[0],'last 12 weeks':[0],'last 4 weeks':[0],'last week':[0],\
                        'next week pessimistic':[0],'next week average':[0],'next week optimistic':[0],\
                        '2 weeks ahead pessimistic':[0],'2 weeks ahead average':[0],'2 weeks ahead optimistic':[0]})
    df2.index = ['{}'.format(category)]
    df2['last 24 weeks'] = round(np.mean(price[-24:]))
    df2['last 12 weeks'] = round(np.mean(price[-12:]))
    df2['last 4 weeks'] = round(np.mean(price[-4:]))
    df2['last week'] = round(np.mean(price[-1:]))
    df2['next week pessimistic'] = price_pess[0]
    df2['next week average'] = price_avg[0]
    df2['next week optimistic'] = price_opt[0]
    df2['2 weeks ahead pessimistic'] = price_pess[1]
    df2['2 weeks ahead average'] = price_avg[1]
    df2['2 weeks ahead optimistic'] = price_opt[1]
    return df2

# Read files with forecast made in R (items, price)
def prepare_forecast(history_path, items_path='items2.csv', price_path='price2.csv'):
    # prepare forecast for each category
    df_items = pd.read_csv(items_path,sep = ';')
    df_price = pd.read_csv(price_path,sep = ';')
    df_history = pd.read_csv(history_path,sep = ',')
    
    fc_list_items = list()
    fc_list_price = list()
    for i in list(df_history.drop_duplicates(['Cat']).Cat):
        fc_list_items.append(prepare_forecast_items(df_history,df_items,i))
        fc_list_price.append(prepare_forecast_price(df_history,df_price,i))
    list_items = pd.concat(fc_list_items)
    list_price = pd.concat(fc_list_price)

    list_items = list_items.ix[:,['last 24 weeks','last 12 weeks','last 4 weeks','last week','next week pessimistic','next week average','next week optimistic','2 weeks ahead pessimistic','2 weeks ahead average','2 weeks ahead optimistic']]
    list_price = list_price.ix[:,['last 24 weeks','last 12 weeks','last 4 weeks','last week','next week pessimistic','next week average','next week optimistic','2 weeks ahead pessimistic','2 weeks ahead average','2 weeks ahead optimistic']]

    next_week = pd.Timestamp(df_items.ix[0,'week']).week
    next_next_week = pd.Timestamp(df_items.ix[1,'week']).week

    list_items.columns = [
        'last 24 weeks',
        'last 12 weeks',
        'last 4 weeks',
        'week {}'.format(next_week - 1),
        'week {} pessimistic'.format(next_week),
        'week {} average'.format(next_week),
        'week {} optimistic'.format(next_week),
        'week {} pessimistic'.format(next_next_week),
        'week {} average'.format(next_next_week),
        'week {} optimistic'.format(next_next_week)
    ]

    list_price.columns = [
        'last 24 weeks',
        'last 12 weeks',
        'last 4 weeks',
        'week {}'.format(next_week - 1),
        'week {} pessimistic'.format(next_week),
        'week {} average'.format(next_week),
        'week {} optimistic'.format(next_week),
        'week {} pessimistic'.format(next_next_week),
        'week {} average'.format(next_next_week),
        'week {} optimistic'.format(next_next_week)
    ]
    return list_items, list_price


# Write to excel file
def write_to_excel(file_name, list_items, list_price, cat):
    writer = pd.ExcelWriter(file_name)
    list_items.to_excel(writer,'Items sold')
    list_price.to_excel(writer,'Paid price')
    writer.save()
    print '{} created'.format(file_name)

# Basic logic
def predict_logic(db, path_input1, path_input2, connection_string, name_output1, name_output2, top_n, year, week, training_weeks, weeks_back, forecast_depth, cat, r_script_path='./arima.r', is_zip=False, is_sql=False, row_sql=''):
    print 'Begin to make a prediction'
    predicted_data = get_predicted_data(path_input1,is_zip = is_zip)
    sales_data = load_sales_data(db=db, path_to_file=path_input2, cat=cat,
                                is_sql=is_sql, connection_string=connection_string)
    joined_data = join_sales_predict(sales_data, predicted_data)
    prepared_data = prepare(joined_data)
    categorizated_data = categorization(prepared_data, file_name=name_output1)
    top_cats = save_top_cat(categorizated_data, prepared_data, top_n, file_name=name_output2)
    
    # Execute R script
    print 'Execute R script'
    import subprocess
    command = 'Rscript'
    path_to_script = './arima.r'
    args = [str(top_n),name_output2,name_output1, str(year), str(week), str(training_weeks), str(weeks_back), str(forecast_depth)]
    cmd = [command, path_to_script] + args
    x = subprocess.check_output(cmd, universal_newlines=True)
    print 'Done'
    
    # Create xlsx
    list_items, list_price = prepare_forecast(name_output2)
    file_name = '{}_forecast.xlsx'.format(cat)
    write_to_excel(file_name, list_items, list_price, cat)
    print 'Done, prediction completed'


if __name__ == '__main__':
    
    # If execute from console

    parser = argparse.ArgumentParser()
    parser.add_argument('-p1','--path1', help='Path to first file', required=True)
    parser.add_argument('-z','--iszip', help='Is file zipped?', required=False)
    parser.add_argument('-p2','--path2', help='Path to second file', required=True)
    parser.add_argument('-s','--issql', help='Is query to db', required=False)
    parser.add_argument('-rs','--r_sql', help='Row SQL', required=False)
    parser.add_argument('-db','--db_path', help='Path to pylitedb', required=False)
    parser.add_argument('-s1','--constr', help='Connection string', required=True)
    parser.add_argument('-o1','--output1', help='Name of first output file', required=True)
    parser.add_argument('-o2','--output2', help='Name of second output file', required=True)
    parser.add_argument('-c','--cat', help='Category', required=True)
    parser.add_argument('-n','--top_n', help='Number of top categories', required=True)
    parser.add_argument('-r','--r_path', help='Path to R script', required=True)
    parser.add_argument('-y','--year', help='Year to R script', required=True)
    parser.add_argument('-w','--week', help='Week to R script', required=True)
    parser.add_argument('-tw','--t_weeks', help='Training weeks to R script', required=True)
    parser.add_argument('-wb','--w_back', help='Weeks back to R script', required=True)
    parser.add_argument('-fd','--f_depth', help='Forecast depth to R script', required=True)
    args = parser.parse_args()
    
    pylitedb_path = args.db_path
    db = Base(pylitedb_path)
    if db.exists():
        db.open()
    else:
        db.create('key')
    df = update_database(db, connection_string)
    
    predict_logic(
        db=db,
        path_input1=args.path1,
        is_zip=args.iszip,
        path_input2=args.path2,
        is_sql=args.issql,
        row_sql=args.r_sql,
        connection_string=args.constr,
        name_output1=args.output1,
        name_output2=args.output2,
        top_n=int(args.top_n),
        cat=args.cat,
        r_script_path = args.r_path,
        year=args.year,
        week= args.week,
        training_weeks=args.t_weeks,
        weeks_back=args.w_back,
        forecast_depth=args.f_depth,
    )
