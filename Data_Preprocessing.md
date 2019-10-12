
## 数据预处理
### 处理非数值型数据


```python
import pandas as pd  

df = pd.DataFrame([ 
                    ['green', 'M', 10.1, 'class1'], 
                    ['red', 'L', 13.5, 'class2'], 
                    ['blue', 'XL', 15.3, 'class1']]) 
df.columns = ['color', 'size', 'price', 'classlabel'] 
print(df)
```

       color size  price classlabel
    0  green    M   10.1     class1
    1    red    L   13.5     class2
    2   blue   XL   15.3     class1


利用pandas将CSV格式的数据读取进来，要将字符串转换为适合机器学习算法训练的数据类型。
先对特征size数据进行处理，由于XL>L>M，size特征有这样的一个顺序特征，所以将这类字符串转换为数值时也该保留顺序特征。利用映射函数map很容易将字符串转换：


```python
size_mapping = {                  
                'XL': 3,                  
                'L': 2,                
                'M': 1} 
df['size'] = df['size'].map(size_mapping) 
print(df)
```

       color  size  price classlabel
    0  green     1   10.1     class1
    1    red     2   13.5     class2
    2   blue     3   15.3     class1


可以利用小技巧将数值数据转换回字符串：


```python
inv_size_mapping = {v: k for k, v in size_mapping.items()}
print(df['size'].map(inv_size_mapping) )
```

    0     M
    1     L
    2    XL
    Name: size, dtype: object


对于分类标签classlabe，只需要将字符串转换为数值特征就好。
利用枚举函数enumerate()和np.unique()将标签导出为映射字典，在用映射函数map将分类标签转换为整数：


```python
import numpy as np
class_mapping = {label:idx for idx,label in                 
                 enumerate(np.unique(df['classlabel']))}
df['classlabel'] = df['classlabel'].map(class_mapping) 
print(df)
```

       color  size  price  classlabel
    0  green     1   10.1           0
    1    red     2   13.5           1
    2   blue     3   15.3           0


同样将整数值可以反映射回分类标签：


```python
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping) 
print(df)
```

       color  size  price classlabel
    0  green     1   10.1     class1
    1    red     2   13.5     class2
    2   blue     3   15.3     class1


也可直接调用sklearn库LabelEncoder类实现：


```python
from sklearn.preprocessing import LabelEncoder 

class_le = LabelEncoder() 
y = class_le.fit_transform(df['classlabel'].values) 
print(y)
```

    [0 1 0]



```python
class_le.inverse_transform(y)
```




    array(['class1', 'class2', 'class1'], dtype=object)



接下来考虑特征color，颜色green，red，blue没有顺序特征，可以采用热编码的方式将其转换，调用pandas中的get_dummies方法转换字符串：


```python
pd.get_dummies(df[['price', 'color', 'size']])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>size</th>
      <th>color_blue</th>
      <th>color_green</th>
      <th>color_red</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>10.1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>13.5</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>15.3</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### 划分数据集为训练集和测试集


```python
import pandas as pd
import numpy as np

df_wine = pd.read_csv('https://archive.ics.uci.edu/'                          
                       'ml/machine-learning-databases/'                          
                       'wine/wine.data', header=None) 
df_wine.columns = ['Class label', 'Alcohol',                  
                       'Malic acid', 'Ash',                
                       'Alcalinity of ash', 'Magnesium',                 
                       'Total phenols', 'Flavanoids',                    
                       'Nonflavanoid phenols',                    
                       'Proanthocyanins',                     
                       'Color intensity', 'Hue',                     
                       'OD280/OD315 of diluted wines',                     
                       'Proline']  
print('Class labels', np.unique(df_wine['Class label'])) 
df_wine.head()
```

    Class labels [1 2 3]





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class label</th>
      <th>Alcohol</th>
      <th>Malic acid</th>
      <th>Ash</th>
      <th>Alcalinity of ash</th>
      <th>Magnesium</th>
      <th>Total phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoid phenols</th>
      <th>Proanthocyanins</th>
      <th>Color intensity</th>
      <th>Hue</th>
      <th>OD280/OD315 of diluted wines</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1</td>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735</td>
    </tr>
  </tbody>
</table>
</div>



首先将数据读取，调用train_test_split函数将数据集划分，参数test_size=0.3表示将数据集的30%
划分给测试集，把70%划分给训练集，stratify=y表示训练集中各个类别的比例和测试集中一样。


```python
from sklearn.model_selection import train_test_split 

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values 
X_train, X_test, y_train, y_test =train_test_split(X, y,                      
                                             test_size=0.3,                      
                                             random_state=0,                     
                                             stratify=y)
```

### 数据正则化
数据标准化对线性模型和梯度下降等优化算法特别有用。但基于树的模型不需要进行标准化。
创建StandardScaler对象，调用fit方法获取样本均值和标准差，再用这些参数去转换测试集。


```python
from sklearn.preprocessing import StandardScaler 

sc = StandardScaler() 
sc.fit(X_train) 
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test) 
```

### 特征的选择
L1正则化：


```python
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

fig = plt.figure() 
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan', 
          'magenta', 'yellow', 'black', 
          'pink', 'lightgreen', 'lightblue', 
          'gray', 'indigo', 'orange'] 
weights, params = [], []
for c in np.arange(-4., 6.): 
    lr = LogisticRegression(penalty='l1', 
                            C=10.**c, 
                            random_state=0,solver='liblinear',multi_class='ovr') 
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1],
             color=color) 
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C') 
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
plt.show() 
```


![Markdown](http://i1.fuimg.com/700703/62a07faa3d644d63.png)


参数C是正则化参数的逆，参数C越小，模型的正则化强度就越强，上述采用L1正则化，可以看到，当C极小的时候，各个特征的权值都趋近于0，然而，当C越大时，有些特征的权重依然徘徊在0附近，这说明对于该分类模型来说，这些特征对于分类的作用没有那么明显，所以，我们可以在训练模型的时候将这些模型去掉，以简化分类模型，减少泛化误差。可以说，L1正则化是一种特征选择技术。


```python
from sklearn.base import clone 
from itertools import combinations 
import numpy as np 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split

class SBS():    
    def __init__(self, estimator, k_features,                  
                 scoring=accuracy_score,                 
                 test_size=0.25, random_state=1):        
        self.scoring = scoring        
        self.estimator = clone(estimator)        
        self.k_features = k_features        
        self.test_size = test_size
        self.random_state = random_state
    
    def fit(self, X, y):                
        X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=self.test_size,                             
                                random_state=self.random_state)            
        dim = X_train.shape[1]        
        self.indices_ = tuple(range(dim))        
        self.subsets_ = [self.indices_]        
        score = self._calc_score(X_train, y_train,                                 
                                 X_test, y_test, self.indices_)        
        self.scores_ = [score]
        while dim > self.k_features:            
            scores = []            
            subsets = []
            for p in combinations(self.indices_, r=dim - 1):                
                score = self._calc_score(X_train, y_train,                                         
                                         X_test, y_test, p)                
                scores.append(score)                
                subsets.append(p)
            best = np.argmax(scores)            
            self.indices_ = subsets[best]            
            self.subsets_.append(self.indices_)            
            dim -= 1
            self.scores_.append(scores[best])        
            self.k_score_ = self.scores_[-1]
        return self
    
    def transform(self, X):        
        return X[:, self.indices_]
    
    def _calc_score(self, X_train, y_train, X_test, y_test,                    
                    indices):        
        self.estimator.fit(X_train[:, indices], y_train)        
        y_pred = self.estimator.predict(X_test[:, indices])        
        score = self.scoring(y_test, y_pred)        
        return score
```


```python
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(knn, k_features=1) 
sbs.fit(X_train_std, y_train) 
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o') 
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy') 
plt.xlabel('Number of features') 
plt.grid() 
plt.show()
```


![Markdown](http://i1.fuimg.com/700703/7309078f2810b069.png)



```python
from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=500, 
                                random_state=1) 
forest.fit(X_train, y_train) 
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):    
    print("%2d) %-*s %f" % (f + 1, 30,                             
                            feat_labels[indices[f]],                             
                            importances[indices[f]])) 
plt.title('Feature Importance') 
plt.bar(range(X_train.shape[1]),         
        importances[indices],         
        align='center')
plt.xticks(range(X_train.shape[1]),           
           feat_labels, rotation=90) 
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
```

     1) Proline                        0.185453
     2) Flavanoids                     0.174751
     3) Color intensity                0.143920
     4) OD280/OD315 of diluted wines   0.136162
     5) Alcohol                        0.118529
     6) Hue                            0.058739
     7) Total phenols                  0.050872
     8) Magnesium                      0.031357
     9) Malic acid                     0.025648
    10) Proanthocyanins                0.025570
    11) Alcalinity of ash              0.022366
    12) Nonflavanoid phenols           0.013354
    13) Ash                            0.013279



![Markdown](http://i1.fuimg.com/700703/dc3f15b157d4100b.png)



```python

```
