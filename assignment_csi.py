import pandas as pd
import numpy as np

X_train = pd.read_csv('https://docs.google.com/uc?id=1--bDvtpHmLmu7JIx5atDZedzkLDe3rzA&export=download')
X_test = pd.read_csv('https://docs.google.com/uc?id=1-2ySFsi0rft_V1lPBNjUswOinHm6uDbk&export=download')

"""СSI - это показатель стабильности эмпирического распределения, основная идея которого - разбить значения переменной на категории и сравнить долю наблюдений, попавших в каждую категорию на тестовой выборке с долей, попавших в эту категорию на бенчмарке (на выборке разработки).
Формула расчета индекса CSI:

\begin{align}
CSI = \sum_{i=1}^K((val_i - dev_i)×ln(\frac{val_i}{dev_i}))
  \end{align}
где K - количество категорий, на которые разбиваются значения переменных, *valᵢ* - доля наблюдений, попавших в категорию i на тестовой выборке, *devᵢ* - доля наблюдений, попавщих в категорию i на выборке разработки.

"""

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
   dev=dev[:-1]
   if(val.min()<dev.min()):
     k.append(val.min())
   if(val.max()>dev.max()):
     k.append(val.max())
   else:
     k.sort()
     k[-1]=dev.max()
   k.sort()
   k[0] -= 0.01
   k[-1] += 0.01
   return k

# Функция для деления на категории по одинаковым расстояниям
def K(val, dev, num_bins=0):
   val=val[~np.isnan(val)]
   dev=dev[~np.isnan(dev)]
   count_dev=len(np.unique(dev))
   if(num_bins==0):
     num_bins=divisor(count_dev)
   min = np.array([val.min(), dev.min()]).min()
   max = np.array([val.max(), dev.max()]).max()
   k = [min + (max - min)*(i)/num_bins for i in range(num_bins+1)]
   k[0] = min - 0.01
   k[-1] = max + 0.01
   return k

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

# Кодирование категорий
def code_categ(values):
  dict_code_categ={}
  i=0
  for j in set(values):
    dict_code_categ[j]=i
    i+=1
  return dict_code_categ

dict_code_categ=code_categ(np.append(X_train['1'].values,X_test['1'].values))
X_test['1'] = X_test['1'].replace(dict_code_categ)
X_train['1'] = X_train['1'].replace(dict_code_categ)

dict_code_categ=code_categ(np.append(X_train['3'].values,X_test['3'].values))
X_test['3'] = X_test['3'].replace(dict_code_categ)
X_train['3'] = X_train['3'].replace(dict_code_categ)

dict_code_categ=code_categ(np.append(X_train['7'].values,X_test['7'].values))
X_test['7'] = X_test['7'].replace(dict_code_categ)
X_train['7'] = X_train['7'].replace(dict_code_categ)

#CSI при делении на равные доли
csi_arr=[]
for i in X_train.columns:
  print('\n\n', i,':')
  csi=CSI(np.array(X_train[i]), np.array(X_test[i]), K_share(np.array(X_train[i]), np.array(X_test[i])))
  csi_arr.append(csi)
  print('CSI=', csi)

#CSI при делении на равные расстояния
csi_arr_K=[]
for i in X_train.columns:
  print('\n\n', i,':')
  csi=CSI(np.array(X_train[i]), np.array(X_test[i]), K(np.array(X_train[i]), np.array(X_test[i])))
  csi_arr_K.append(csi)
  print('CSI=', csi)

print(csi_arr)  #CSI при делении на равные доли
print(csi_arr_K)#CSI при делении на равные расстояния
