# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter

#Функция для поиска индекса по значению или значениям (в таком случае выдает массив с индексами каждого эл-нта)
#На вход подается массив, массив значений или одно значение и необязательный параметр, по которому определяется ось индекса
#Возвращает массив индексов или индекс
def find_index(array, text, axis='col'):
   #Если на входе массив значений, то обрабатывается каждый эл-нт отдельно
   if(((type(text)==list)|(type(text)==np.ndarray))):   
      index=list(map(lambda i: find_index(array, i), text))
      #Если массив с найденными индексами одномерный,
      if((np.array(index).shape[0]==1)|(len(np.array(index).shape)>1)): 
         #убирается лишняя размерность
         if(np.array(index).shape[1]==1):                              
            index=list(map(lambda i: find_index(array, i)[0], text))
            return index
         #Иначе выводится без преобразований
         else:            
            return index      
      else:
         return index
   #Если на входе не массив, функция преобразует всё в строки и ищет нужный индекс
   else:                                              
     try:
        array=np.array(array).astype(str)
        text=str(text)
        index=np.array(list(map(lambda i: list(i), np.where(array==text)))).T.tolist()
        if (np.array(index).shape[0]==1):
           try:
              if(np.array(index).shape[1]==1):
                 return index[0][0]
              else:
                 return index
           except:
              return index[0]
        elif (len(index)==0):
           print('Значение', text,'не найдено', '\n')
           return None
        else:
            return index
     except Exception as err:
        print('Ошибка в функции find_index', err, '\n')
        return None

#Функция для поиска мультииндекса, работает с двумя колонками, для большего числа требуется доработка ?
def find_index_multicolum(array, array_text):
    #Если на входе массив, ищем мультииндекс, иначе-просто индекс
    if((type(array_text)==list)|(type(array_text)==np.ndarray)):
        try:
            #Если в массиве все элемены одинаковы, подсчитываем самый частый индекс строки
            if(len(set(array_text))==1):
                print('Одинаковые значения при поиске мультииндекса', '\n')
                #Стоит искать один, если они одинаковы?
                array_text=array_text[0]
                row_after_find=find_index(array, array_text)
                #Берем только индексы строк
                row_after_find=np.array(row_after_find)[:,0]
                #Подсчитываем значения
                diction=Counter(row_after_find)
                #Сортируем и берем индексы, число к. максимально
                arr=np.array(sorted(diction.items(), key=lambda item: item[1]))
                max_count=max(arr[:,1])
                index_row=arr[arr[:,1]==max_count, 0]
            #Иначе - выводим индекс, являющийся пересечением двух мн-ств индексов строк
            else:
                arr_after_find=np.array(find_index(array, array_text))
                def array_after_find_to_set(arr_after_find_i):
                    elem_after_find=np.array(arr_after_find_i)
                    try:
                        if(elem_after_find.shape[1]==1):
                            elem_after_find=elem_after_find[0]
                        else:
                            elem_after_find=elem_after_find[:,0]
                        return set(elem_after_find)
                    except:
                        elem_after_find=elem_after_find[0]
                        return elem_after_find
                np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
                sets_after_find=list(map(lambda i: array_after_find_to_set(i), arr_after_find))
                #Выбираем изначальное множество и оставляем пересечение всех в index_row
                index_row=set()
                try:
                    index_row.add(sets_after_find[0])
                except:
                    index_row.update(sets_after_find[0])

                for i in sets_after_find[1:]:
                    index_row=index_row.intersection(i)
                print('Мультииндекс =', index_row, 'Минимальный индекс', min(index_row), '\n')

            if (len(index_row)>1):
                print('Несколько значений, результат - минимальный индекс', '\n')
                return min(index_row)
            else:
                return min(index_row)
        except Exception as err:
            print('Ошибка в функции find_index_multicolum', err, '\n')
            return None
    else:
        return find_index(array, array_text)

        
#Сортировка массива по одному индексу или по нескольким, заданных как одномерный массив
#На вход подается массив, одно значение и необязательный параметр - тип сортировки
#Возвращает отсортированный numpy массив
def sort(array, column, ascending=True):
    if((type(column)==list)|(type(column)==np.ndarray)):
        array=np.array(array)
        column=np.flip(column, 0)
        ind = np.lexsort(list(map(lambda i: array[:,i], column)))
        array=array[ind]
        return array
    else:
        try:
            if (ascending==True):
                array = array[array[:, column].argsort()]
            else:
                array = array[array[:, column].argsort()[::-1]]
            return array
        except Exception as err:
            print('Ошибка в функции sort', err)
            print('Индекс', column, "не найден", '\n')
            return None   

#Ф-ция, которая сортирует массив по колонкам
#На вход подается массив, массив значений в порядке важности сортировки и необязательный параметр - тип сортировки
def sort_columns(array, columns, ascending=True):
        if (ascending==True):
            sort_col=columns.argsort()
            columns=columns[sort_col]
            array = array[:, sort_col]
        else:
            sort_col=columns.argsort()
            columns=columns[sort_col[::-1]]
            array = array[:, sort_col[::-1]]
        return array

# группировка массива по column
#Подается на вход numpy array или массив python (list), массив индексов или индекс, по которым будет группировка и необязательный параметр флаг удаления столбцов, по которым была группировка
#возвращает массив numpy с массивами, содержащие сгруппированные записи, и массив с значениями column
#т.е. индекс сгруппированной записи будет равен индексу значения в col_after_group_by, по которому записи сгруппированы
#к сгруппированным записям можно обратиться array_group_by[find_index_multicolum(col_after_group_by, "нужное значение column")]
def group_by(array, column, flag_delete=True):
    array=sort(np.array(array), column)
    col_after_group_by, col_after_group_by_index =np.unique(array[:, column].astype("<U22"), return_index=True, axis=0)
    if(flag_delete):
        array_group_by=np.split(np.delete(array, column, axis=1), col_after_group_by_index[1:])
    else:
        array_group_by=np.split(array, col_after_group_by_index[1:])
    return np.array(array_group_by), np.array(col_after_group_by)

#Применение функций sum, mean и тд к группе и запись значения в качестве нового столбца
#Data_group_by-сгруппированные данные, Col_for_func-индексы строк дя применения функции, Function - передача функциибез ()
def Function_after_group_by(Data_group_by, Col_for_func, Function):
    return np.array(list(map(lambda x: np.concatenate([x,np.array([Function(x[:,Col_for_func])]*len(x))[None,:].T], axis=1), Data_group_by)))

#Функция для группировки и применения нескольких агрегации 
#На вход подаются numpy array или массив, массив индексов или индекс для группировки
def group_aggregation(Data, Col_for_group, Col_for_func, Function, return_flag='group'):
    Data_group_by, _ =group_by(np.array(Data), Col_for_group, flag_delete=False)
    print('Col_for_func и Function в функции для агрегации', Col_for_func, Function, '\n')
    Col_for_func=np.array(Col_for_func)
    Function=np.array(Function)
    
    Data_aggr=Data_group_by
    for i in range(len_rows(Col_for_func)):
        Data_aggr= Function_after_group_by(Data_aggr, Col_for_func[i], Function[i])
    if return_flag=='group':
        return Data_aggr
    else:
        Data_aggr=np.vstack(Data_aggr)
        return Data_aggr

"""
#Ф-ция для представления "длинной таблицы" в "широкую" Нереализован вариант, если колонки в группах не совпадают
#index_for_col - индекс колонки, которая будет наименованием колонок в широкой таблице
#names_for_value - наименования колонок, которые будут значениями
#index_for_group - индекс колонки, которая будет делить значения в группировке в широкой таблице; 
#names_for_row - список значений, к. будет в табл., по умолч. все
#Возвращает широкую таблицу, наименование колонок и наименование строк
def to_wide(Data, Data_col, names_for_value, index_for_col, index_for_group, names_for_row='all'):
    Data_row=np.array([])
    indexes_for_value= find_index(Data_col, names_for_value)
    if (names_for_row=='all'):
        names_for_row=np.unique(Data[:, index_for_group])
    if((type(names_for_row)==list)|(type(names_for_row)==np.ndarray)):
        Data_res=[]
        Data_col=[]
        for name in names_for_row:
            Data_group, _ =group_by(Data, [index_for_group, index_for_col], flag_delete=False)
            Data_to_wide=np.array(list(map(lambda i: i[i[0][index_for_group]==name], Data_group)))
            Data_to_wide=np.array(list(map(lambda i: i[:,1,
                                        [index_for_col]+indexes_for_value], Data_to_wide)))
            Data_to_wide=np.array([x[0] for x in Data_to_wide if x.any()])
            Data_to_wide=Data_to_wide.T
            Data_res.append(Data_to_wide[1:])
            Data_col.append(Data_to_wide[0])
            Data_row=np.concatenate([Data_row, list(map(lambda x: x+'.'+name, names_for_value))], axis=0)
        Data_res=np.vstack(Data_res)
        Data_col=np.array(Data_col)
        sets=list(map(lambda x: set(x), Data_col))
        cond=sets[0]
        for i in sets[1:]:
            cond=cond.difference(i)
        if(cond==set()):
            #Добавление колонки имен (name) в начало таблицы
            #Data_res=np.array(list(map(lambda x: 
            #    np.concatenate([np.array([names_for_row[x]]*len_rows(Data_res[0]))[None,:].T, 
            #                    Data_res[x]],axis=1), range(len(names_for_row)))))
            #Data_res=np.concatenate(Data_res, axis=1)
            return Data_res, Data_col[0], Data_row
        #else...Когда таблицы с разными именами имеют разные колонки..
    
    """"""else:
        Data_group, _ =group_by(Data, [index_for_group,index_for_col], flag_delete=False)
        Data_to_wide=np.array(list(map(lambda i: i[i[0][index_for_group]==names_for_row], Data_group)))
        Data_to_wide=np.array(list(map(lambda i: i[:,1,
                                    [index_for_col]+indexes_for_value], Data_to_wide)))
        Data_to_wide=np.array([x[0] for x in Data_to_wide if x.any()])
        Data_to_wide=Data_to_wide.T
        Data_res=Data_to_wide[1:]
        Data_col=np.array(Data_to_wide[0].tolist())
        Data_res=np.concatenate([np.array([names_for_row]*len_rows(Data_res))[None,:].T, Data_res], axis=1)
        return Data_res, Data_col  """
    
#Функция для фильтра по дате в типе "строка"
#На вход подается numpy array или list, индекс колонки с датой, минимальная или максимальная дата для фильтра, формат даты, необяхательный параметр - значение указанной даты для фильтра, по-умолч. минимальная дата
def filter_by_min_date(data, column, date_min, format_date_min, ascending='min'):
    date_min=datetime.strptime(date_min, format_date_min)
    data_=np.array(data)
    if ascending=='min':
      return data_[data_[:,column]>=date_min]
    else:
      return data_[data_[:,column]<=date_min]

#
#Функция для преобразования столбцов numpy array или list в тип datetime
#На вход подается numpy array или list, формат и массив индексов или индекс колонки с датой, если не указано преобразует все колонки
#Возврашает numpy array
def str_to_date(array, format_, columns=['all']):
    array_=np.array(array)
    def parser_date(string, format_):
        try:
            return datetime.strptime(string, format_)
        except:
            return string
    if (columns==['all']):
        columns=np.arange(len_columns(array_))
    for j in columns:
        array_[:, j]=np.array(list(map(lambda i: parser_date(i, format_), array_[:, j])))
    return array_

def len_columns(array):
    return array.shape[1]

def len_rows(array):
    return array.shape[0]

#drop_na (удаление строки, если есть nan в указанных колонках)
#На вход подается numpy array или list и массив индексов или индекс колонки с датой, если не указано удаляет во всех колонках
#Возврашает numpy array
def drop_na(array, columns=['all']):                            
    if (columns==['all']):
        columns=np.arange(len_columns(array))
    return np.delete(array, np.unique(np.where(((array[:,columns]!=array[:,columns])|(array[:,columns]==None)|
                                             (array[:,columns]=='NAN')|(array[:,columns]=='nan')|
                                             (array[:,columns]=='None')|(array[:,columns]=='Nan')|
                                                (array[:,columns]=='NaN')|(array[:,columns]=='Null')|
                                                (array[:,columns]=='NULL')))[0]),  axis=0)

#Функция для сэмплирования, где из каждой группы берется случайная запись
#На вход подается  numpy array или list и массив индексов или индекс для группировки
#
def sampling(array, column):#, ObsDate):
   np.random.seed(29)
   array_group_by, col_after_group_by=group_by(array, column, flag_delete=False)
   #array_group_by=np.array(list(map(lambda x: sort(x, column+[ObsDate],ascending=False)[:12], array_group_by)))
   random_row=list(map(lambda x: x[np.random.choice(x.shape[0])], array_group_by))
   print('Строки в сэмплировании', random_row[:2], '\n', 'Кол-во строк', len(random_row), 'Кол-во индексов', len(col_after_group_by), '\n')
   return random_row   
   
#Функция для скользящего среднего
#На вход подается numpy array или list, размер окна и направление окна от первого элемета - 'right', 'left' или по середине(любое другое значение в параметре)
#Возврашает numpy array
def moving_avg(a, n=3, align='right'):
   ret = np.cumsum(a, dtype=float)
   ret[n:] = ret[n:] - ret[:-n]
   res=np.array(ret[n - 1:] / n)
   if align=='right':
       res=np.concatenate([[None]*(len(a)-len(res)), res], axis=0)
   elif align=='left':
       res=np.concatenate([res, [None]*(len(a)-len(res))], axis=0)
   else:
       res=np.concatenate([[None]*(int((len(a)-len(res))/2)), res, [None]*(int((len(a)-len(res))/2))], axis=0)
   return  res


# Функция для поиска наибольшего делителя, не большее, чем max_block
def divisor(n, max_block=24): 
    i=2
    pre_i=1
    while((i<max_block)&(n>=i)): 
      while((n%i!=0)&(n>=i)&(i<max_block)):
        i+=1
      if(n%i==0):
        pre_i=i
        i+=1
    if(n%i==0):
      return i
    else:
      if(pre_i==1):
        return divisor(n+1)
      else:
        return pre_i

    
# Функция для деления на категории по одинаковым долям
def K_share(val, dev, num_bins=0):
   val=val[~np.isnan(val)]
   dev=dev[~np.isnan(dev)]
   
   count_dev=len(np.unique(dev))
   dev=np.append(dev,10**100)
   dev.sort()
   if(num_bins==0):
     num_bins=divisor(count_dev)
   k = [np.unique(dev)[i] for i in range(0, count_dev+1, int(count_dev/num_bins))]
   dev=dev[:-2]
   
   print(k)
   if(len(k)<=3):
      k[-1]=np.array(k).mean()
   else:
      k[-1]=dev[-1]

   k.sort()
   k[0] -= 0.01
   k[-1] += 0.01
   print(k)
   return k

# Функция для деления на категории по одинаковым расстояниям
def K(val, dev, num_bins=0):
   val=val[~np.isnan(val)]
   dev=dev[~np.isnan(dev)]
   count_dev=len(np.unique(np.append(dev,val)))
   if(num_bins==0):
     num_bins=divisor(count_dev)
   min = np.array([val.min(), dev.min()]).min()
   max = np.array([val.max(), dev.max()]).max()
   k = [min + (max - min)*(i)/num_bins for i in range(num_bins+1)]
   k[0] = min - 0.01
   k[-1] = max + 0.01
   return k


# Кодирование категорий для pandas
# Пример: 
#dict_code_categ=code_categ(np.append(X_train['3'].values,X_test['3'].values))
#X_test['3'] = X_test['3'].replace(dict_code_categ)
def code_categ(values):
  dict_code_categ={}
  i=0
  for j in set(values):
    dict_code_categ[j]=i
    i+=1
  return dict_code_categ

#Функция для теста DownTurn
#На вход numpy массив, массив с названием его столбцов, название типа результата, по которым будет группировка и название столбца с датой
#На выходе датафреим с мультииндексом [агрегация, Result(тип результата)]
def calculate_DT(Data, Data_col_in_f, Result, Month):
    Data=Data.copy()
    Data_col_in_f=Data_col_in_f.copy()
    #Определение типов моделей в данных
    type_models=np.unique(Data[:,find_index(Data_col_in_f, Result)])
    #группировка по месяцу дефолта и исходу NFL   
    Data_group, _ =group_by(Data, find_index(Data_col_in_f, [Result,Month]), flag_delete=False)
    Data_aggr=group_aggregation(Data, find_index(Data_col_in_f, [Result,Month]),
                [find_index(Data_col_in_f, Target),find_index(Data_col_in_f, DT), find_index(Data_col_in_f, Target)], [len,max,sum])
    
    print('shape Data_aggr', Data_aggr.shape)
    Data=np.array(list(map(lambda x: x[0], Data_aggr)))
    print('shape Data', Data.shape)
    print('Data[:2]', Data[:2], '\n')

    Data_col=Data_col_in_f+['Obs']+['Max_DT']+['Sum_Target']
    print('Размер Data после добавления агрегаций:', Data.shape)
    print('Размер Data_col после добавления агрегаций:', len(Data_col), '\n')
    
    #Преобразование в широкую таблицу
    Data=Data[:,find_index(Data_col, [Result, Month, 'Obs','Max_DT','Sum_Target'])]
    Data_col_res=[Result, Month, 'Obs','Max_DT','Sum_Target']
    Data=pd.DataFrame(Data, columns=Data_col_res)
    
    Data_wide=pd.pivot(Data, index=Month, columns=Result, values=['Obs','Max_DT','Sum_Target']).T
    Data_wide.index.names=['Func', 'fact_result_for_monitoring']
    Data_wide=Data_wide.fillna(0)
    
    print('Размер широкой таблицы Data_wide', Data_wide.shape)
    print('Data_wide', Data_wide.iloc[:,:2], '\n')
    
    """
    Пример мультииндекса
    MultiIndex([(       'Obs',     'Full Loss'),
                (       'Obs', 'Non-full loss'),
                (    'Max_DT',     'Full Loss'),
                (    'Max_DT', 'Non-full loss'),
                ('Sum_Target',     'Full Loss'),
                ('Sum_Target', 'Non-full loss')],
    """


    #Добавление среднего target по всем группам Result
    
    
    Series=sum(list(map(lambda x: Data_wide.loc['Sum_Target',x], type_models)))/sum(list(map(lambda x: Data_wide.loc['Obs',x], type_models)))
    
    def append_with_multiindex(Data, New_index, Values):
        index_names=Data.index.names
        Data=Data.reset_index()
        Data.loc[len(Data)]=np.concatenate([New_index, Values], axis=0)
        Data=Data.set_index(index_names)
        return Data
    
    Data_wide=append_with_multiindex(Data_wide, ['Full Avg Target', 'All'], Series)
    print('Data_wide[:,:1] после добавления Full Avg Target', '\n', Data_wide.iloc[:,:1], '\n')
        
    #скользящее среднее по средним потерям в модуле NFL
    Data_means=moving_avg(Data_wide.loc['Full Avg Target', 'All'].values, n=DTPeriod, align='right')
    print('Data_means', Data_means, '\n')
    Data_wide=append_with_multiindex(Data_wide, ['Moving Means Full Avg Target', 'All'], Data_means)
    print('Data_wide[:,:-1] после добавления Moving Means Full Avg Target', '\n', Data_wide.iloc[:,-1], '\n')

    Data_dtend = max(Data_wide.loc['Moving Means Full Avg Target'])
   # Data_dtend = max(Data_wide.loc['Moving Means Full Avg Target', ~Data_wide.loc['Moving Means Full Avg Target', 'All'].isna()])
    Data_dtstart = Data_dtend + relativedelta(months=(1-DTPeriod))
    print('Дата начала окна', Data_dtstart)
    print('Дата окончания окна', Data_dtend, '\n')
    Data_on_window_date=Data_wide.loc[:,(Data_wide.columns<=Data_dtend) & (Data_wide.columns>=Data_dtstart)]
    
    Data_full_loss_avg=[]
    Data_full_loss_avgDT=[]
    Data_DT_mon=[]
    
    for type_model in type_models:
        Data_full_loss_avg.append(sum(Data_wide.loc['Sum_Target',type_model])/sum(Data_wide.loc['Obs', type_model]))
        Data_full_loss_avgDT.append(sum(Data_on_window_date.loc['Sum_Target',type_model])/sum(Data_on_window_date.loc['Obs',type_model]))
        Data_DT_mon.append(max(0,Data_full_loss_avgDT[-1] - Data_full_loss_avg[-1]))
    return Data_wide, [[Data_dtstart]*len(type_models)], [[Data_dtend]*len(type_models)], type_models, Data_full_loss_avg, Data_full_loss_avgDT, Data_DT_mon

#Функция для расчета теста CSI, иногда его еще называют PSI - population stability index 
#
#
def CSI(val, dev, k):
  def calc_csi(val_i, dev_i):     #расчет CSI
    if(dev_i==0):
      dev_i=10**(-4)
    if(val_i==0):
      val_i=10**(-4)
    return ((val_i - dev_i) * np.log(val_i / dev_i))
 
  def count_i(X, start, end):  #Кол-во элементов, входящих в интервал
    i=0
    count_i=0
    while (X[i]<end):
      if(X[i]>=start):
        count_i+=1
      if(i<len(X)-1):
        i+=1
      else:
        return count_i
    return count_i

  sum_val_i=0
  sum_dev_i=0
  count_val=len(val)
  count_dev=len(dev)
  csi=np.array([])

  val_nan=len(val[np.isnan(val)])/count_val
  dev_nan=len(dev[np.isnan(dev)])/count_dev
  csi=np.append(csi, calc_csi(val_nan, dev_nan))

  val=val[~np.isnan(val)]
  dev=dev[~np.isnan(dev)]
  val.sort()
  dev.sort()

  for i in range(len(k)-1):
    val_i=count_i(val, k[i], k[i+1])/count_val
    sum_val_i+=val_i
    dev_i=count_i(dev, k[i], k[i+1])/count_dev
    print('Доля на тесте val_'+str(i),'{:<20.2%}'.format(val_i), 'Доля на разработке dev_'+str(i), '{:.2%}'.format(dev_i))
    sum_dev_i+=dev_i
    csi=np.append(csi, calc_csi(val_i, dev_i))
    
  print('{:<46}'.format('\nСумма не пустых на тесте sum_val_i ='), '{:.2%}'.format(sum_val_i))
  print('{:<45}'.format('Сумма не пустых на разработке sum_dev_i ='), '{:.2%}'.format(sum_dev_i))
  print('{:<45}'.format('Сумма пустых на тесте val_nan ='),  '{:.2%}'.format(val_nan))
  print('{:<45}'.format('Сумма пустых на разработке dev_nan ='),  '{:.2%}'.format(dev_nan))
  return csi.sum()


#CSI при делении на равные доли
def CSI_equal_shares(X_train, X_test):
  csi_arr=[]
  for i in X_train.columns:
    print('\n\n', i,':')
    csi=CSI(np.array(X_train[i]), np.array(X_test[i]), K_share(np.array(X_train[i]), np.array(X_test[i])))
    csi_arr.append(csi)
    print('CSI=', csi)
  return csi_arr

#CSI при делении на равные расстояния
def CSI_equal_bins(X_train, X_test):
  csi_arr_K=[]
  for i in X_train.columns:
    print('\n\n', i,':')
    csi=CSI(np.array(X_train[i]), np.array(X_test[i]), K(np.array(X_train[i]), np.array(X_test[i])))
    csi_arr_K.append(csi)
    print('CSI=', csi)
  return csi_arr_K

#Функции для расчета теста Loss_Shortfall
def Loss_Shortfall(data, Target, i, EAD):
   #можно еще транспонировать и взять колонки как строки?
   array = np.array(data[:, [Target, i, EAD]]) 
   array=drop_na(array)
   if(len(array) == 0):
      print('Пустой массив для Loss_Shortfall', '\n')
      return([None, ''])
   #print('Сумма реальных значений', np.sum(array[:,0]))
   #print('Сумма предсказанных значений', np.sum(array[:,1]))
   #print('Сумма переменной EAD', np.sum(array[:,2]))
  
   try:
      loss_shortfall = 1 - np.sum(array[:, 1] * array[:,2]) / np.sum(array[:, 0] * array[:,2])
   except ZeroDivisionError:
      print('ZeroDivisionError in ', i)
      loss_shortfall=None
      grade=None
   if((loss_shortfall >= -0.1)&(loss_shortfall <= 0)):
      grade='G'
   elif((loss_shortfall >= -0.2)&(loss_shortfall < -0.1)):
      grade='A'
   else:
      grade='R'  

   return [str(loss_shortfall), grade, str(len_rows(data))]

#Для вычисления Loss_Shortfall с разными данными, возвращает массив значений теста, grade и размер выборки
def Loss_Shortfall_calculate(Data_with_filter, Target, ModelChars, EAD, type_Data='Data'):
   lshmass=[]
   ModelChars_i=[]
   for i in ModelChars:
      lshmass_i=Loss_Shortfall(Data_with_filter, find_index(Data_col, Target),find_index(Data_col, i), find_index(Data_col, EAD))
      lshmass.append(lshmass_i)
      ModelChars_i.append(i)
      print('Значение теста LossShortFall, Grade, Размер выборки', lshmass_i)
   print('\n','lshmass', type_Data,  lshmass, '\n')
   return lshmass, ModelChars_i

def power_stat_gr(data, Target, predict_i):
    array = np.array(data[:, [Target, predict_i]]) 
    array=drop_na(array)
    n=len_rows(array)
    if(n>0):
        #Сортируем по убыванию predict_i
        array=sort(array, 1, ascending=False) 
        
        #Группируем по predict_i и по группе считаем средний Target, записывая как новую колонку в массив
        array_group_by, col_after_group_by=group_by(array, 1, flag_delete=False)
        array_with_avg=np.array(list(map(lambda x: np.column_stack((x, [np.mean(np.array(x)[:,0])]*len_rows(np.array(x)))), array_group_by)))
        
        #Разгруппировываем
        array_with_avg=np.vstack(array_with_avg)
        #print('len array_with_avg', len(array_with_avg))
        
        array_with_avg=sort(array_with_avg, 1, ascending=False) 
        predicted_relative_sum = np.cumsum(array_with_avg[:,2]) / sum(array_with_avg[:,2])
        print('predicted_relative_sum',predicted_relative_sum)
        
        predicted_weights=np.arange(1,n+1) / ([n]*n)
        actual_weights=predicted_weights
        print('predicted_weights',predicted_weights)
        
        df_ = sort(array, 0, ascending=False)
        actual_relative_sum = np.cumsum(df_[:,0]) / sum(df_[:,0])
        
        gini_predicted = 0.5 * sum((predicted_weights[2:n]-predicted_weights[1:n-1])*(predicted_relative_sum[2:n]+predicted_relative_sum[1:n-1])) - (0.5*sum((predicted_weights[2:n]-predicted_weights[1:n-1])*(predicted_weights[2:n]+predicted_weights[1:n-1])))
        gini_actual = 0.5*sum((actual_weights[2:n]-actual_weights[1:n-1])*(actual_relative_sum[2:n]+actual_relative_sum[1:n-1])) - (0.5*sum((actual_weights[2:n]-actual_weights[1:n-1])*(actual_weights[2:n]+actual_weights[1:n-1])))   
        print('gini_predicted',gini_predicted)
        print('gini_actual',gini_actual, '\n')
        
        result = gini_predicted/gini_actual
    else:
        print('Пустой массив для PowerStat', '\n')
        result = None
    return result

#Для вычисления PowerStat с разными данными, возвращает массив значений теста, grade и размер выборки
def power_stat_calculate(Data_with_filter, Target, ModelChars, compscore, allscore, modulescore, vars_, type_Data='Data'):
    psmass=[]
    ModelChars_i=[]
    for i in ModelChars:
        ModelChars_i.append(i)
        try:
            power_stat_value=power_stat_gr(Data_with_filter, find_index(Data_col, Target),find_index(Data_col, i))
            
            if(power_stat_value!=None):
                  
                    if((i in compscore)|(i in allscore)):
                        if(power_stat_value >= 0.30):
                            power_stat_grade = 'G'
                        elif((power_stat_value >= 0.15)&(power_stat_value < 0.30)):
                            power_stat_grade = 'A'
                        else:
                            power_stat_grade = 'R'

                    if((i in modulescore)|(i in vars_)):
                        if(power_stat_value >= 0.25):
                            power_stat_grade = 'G'
                        elif((power_stat_value >= 0.10)&(power_stat_value < 0.25)):
                            power_stat_grade = 'A'
                        else:
                            power_stat_grade = 'R'
   
        except ZeroDivisionError:
            power_stat_value=None
            power_stat_grade='-'
            print('Деление на ноль при подсчете теста PowerStat', '\n')
          
        psmass.append([power_stat_value, power_stat_grade, len_rows(Data_with_filter)])
        print('Значение теста PowerStat, grade, obs_cnt]', psmass[-1])
    print('\n','psmass', type_Data,  psmass, '\n')
    return psmass, ModelChars_i

