# 数据分析DAY04

### 加载文件

numpy提供了函数用于加载逻辑上可被解释为二维数组的文本文件，格式如下：

```
数据项1 <分隔符> 数据项2 <分隔符> ... <分隔符> 数据项n
例如：
AA,AA,AA,AA,AA
BB,BB,BB,BB,BB
...
或：
AA:AA:AA:AA:AA
BB:BB:BB:BB:BB
...


```

调用numpy.loadtxt()函数可以直接读取该文件并且获取ndarray数组对象：

```python
import numpy as np
# 直接读取该文件并且获取ndarray数组对象 
# 返回值：
#     unpack=False：返回一个二维数组
#     unpack=True： 多个一维数组
np.loadtxt(
    '../aapl.csv',			# 文件路径
    delimiter=',',			# 分隔符
    usecols=(1, 3),			# 读取1、3两列 （下标从0开始）
    unpack=False,			# 是否按列拆包
    dtype='U10, f8',		# 制定返回每一列数组中元素的类型
    converters={1:func}		# 转换器函数字典
)    

```

案例：读取aapl.csv文件，得到文件中的信息：

```python
import numpy as np
import datetime as dt
# 日期转换函数
def dmy2ymd(dmy):
	dmy = str(dmy, encoding='utf-8')
	time = dt.datetime.strptime(dmy, '%d-%m-%Y').date()
	t = time.strftime('%Y-%m-%d')
	return t
dates, opening_prices,highest_prices, \
	lowest_prices, closeing_pric es  = np.loadtxt(
    '../data/aapl.csv',		# 文件路径
    delimiter=',',			# 分隔符
    usecols=(1, 3, 4, 5, 6),			# 读取1、3两列 （下标从0开始）
    unpack=True,
    dtype='M8[D], f8, f8, f8, f8',		# 制定返回每一列数组中元素的类型
    converters={1:dmy2ymd})

```

案例：使用matplotlib绘制K线图

1. 绘制dates与收盘价的折线图：

```python
import numpy as np
import datetime as dt
import matplotlib.pyplot as mp
import matplotlib.dates as md

# 绘制k线图，x为日期
mp.figure('APPL K', facecolor='lightgray')
mp.title('APPL K')
mp.xlabel('Day', fontsize=12)
mp.ylabel('Price', fontsize=12)

#拿到坐标轴
ax = mp.gca()
#设置主刻度定位器为周定位器（每周一显示主刻度文本）
ax.xaxis.set_major_locator( md.WeekdayLocator(byweekday=md.MO) )
ax.xaxis.set_major_formatter(md.DateFormatter('%d %b %Y'))
#设置次刻度定位器为日定位器 
ax.xaxis.set_minor_locator(md.DayLocator())
mp.tick_params(labelsize=8)
dates = dates.astype(md.datetime.datetime)

mp.plot(dates, opening_prices, color='dodgerblue',
		linestyle='-')
mp.gcf().autofmt_xdate()
mp.show()


```

1. 绘制每一天的蜡烛图：

```python
#绘制每一天的蜡烛图
#填充色：涨为白色，跌为绿色
rise = closeing_prices >= opening_prices
color = np.array([('white' if x else 'limegreen') for x in rise])
#边框色：涨为红色，跌为绿色
edgecolor = np.array([('red' if x else 'limegreen') for x in rise])

#绘制线条
mp.bar(dates, highest_prices - lowest_prices, 0.1,
	lowest_prices, color=edgecolor)
#绘制方块
mp.bar(dates, closeing_prices - opening_prices, 0.8,
	opening_prices, color=color, edgecolor=edgecolor)

```

### 算数平均值

```
S = [s1, s2, ..., sn]


```

样本中的每个值都是真值与误差的和。

```
算数平均值：
m = (s1 + s2 + ... + sn) / n


```

算数平均值表示对真值的无偏估计。

```python
m = np.mean(array)
m = array.mean()

```

案例：计算收盘价的算术平均值。

```python
import numpy as np
closing_prices = np.loadtxt(
    '../../data/aapl.csv', delimiter=',',
    usecols=(6), unpack=True)
mean = 0
for closing_price in closing_prices:
    mean += closing_price
mean /= closing_prices.size
print(mean)
mean = np.mean(closing_prices)
print(mean)

```

### 加权平均值

样本：$S = [s_1, s_2, s_3 ... s_n]$

权重：$W =[w_1, w_2, w_3 ... w_n]$

加权平均值：$a = \frac{s_1w_1 + s_2w_2 + ... + s_nw_n}{w_1+w_2+...+w_n}$

```python
a = np.average(closing_prices, weights=volumes)
```



VWAP - 成交量加权平均价格（成交量体现了市场对当前交易价格的认可度，成交量加权平均价格将会更接近这支股票的真实价值）

```python
import numpy as np
closing_prices, volumes = np.loadtxt(
    '../../data/aapl.csv', delimiter=',',
    usecols=(6, 7), unpack=True)
vwap, wsum = 0, 0
for closing_price, volume in zip(
        closing_prices, volumes):
    vwap += closing_price * volume
    wsum += volume
vwap /= wsum
print(vwap)
vwap = np.average(closing_prices, weights=volumes)
print(vwap)

```

TWAP - 时间加权平均价格（时间越晚权重越高，参考意义越大）

```python
import datetime as dt
import numpy as np

def dmy2days(dmy):
    dmy = str(dmy, encoding='utf-8')
    date = dt.datetime.strptime(dmy, '%d-%m-%Y').date()
    days = (date - dt.date.min).days
    return days

days, closing_prices = np.loadtxt(
    '../../data/aapl.csv', delimiter=',',
    usecols=(1, 6), unpack=True,
    converters={1: dmy2days})
twap = np.average(closing_prices, weights=days)
print(twap)

```

 

### 最值

**np.max()  np.min() np.ptp()：** 返回一个数组中最大值/最小值/极差

```python
import numpy as np
# 产生9个介于[10, 100)区间的随机数
a = np.random.randint(10, 100, 9)
print(a)
print(np.max(a), np.min(a), np.ptp(a))
```

**np.argmax() np.argmin()：** 返回一个数组中最大/最小元素的下标

```python
print(np.argmax(a), np.argmin(a))
```

**np.maximum() np.minimum()：** 将两个同维数组中对应元素中最大/最小元素构成一个新的数组

```python
print(np.maximum(a, b), np.minimum(a, b), sep='\n')
```

案例：评估AAPL股票的波动性。

```python
import numpy as np
highest_prices, lowest_prices = np.loadtxt(
    '../../data/aapl.csv', delimiter=',',
    usecols=(4, 5), dtype='f8, f8', unpack=True)
max_price = np.max(highest_prices)
min_price = np.min(lowest_prices)
print(min_price, '~', max_price)
```

查看AAPL股票最大最小值的日期，分析为什么这一天出现最大最小值。

```python
import numpy as np
dates, highest_prices, lowest_prices = np.loadtxt(
    '../../data/aapl.csv', delimiter=',',
    usecols=(1, 4, 5), dtype='U10, f8, f8',
    unpack=True)
max_index = np.argmax(highest_prices)
min_index = np.argmin(lowest_prices)
print(dates[min_index], dates[max_index])
```

观察最高价与最低价的**波动范围**，分析这支股票底部是否坚挺。  

```python
import numpy as np
dates, highest_prices, lowest_prices = np.loadtxt(
    '../../data/aapl.csv', delimiter=',',
    usecols=(1, 4, 5), dtype='U10, f8, f8',
    unpack=True)
highest_ptp = np.ptp(highest_prices)
lowest_ptp = np.ptp(lowest_prices)
print(lowest_ptp, highest_ptp)
```

### 中位数

将多个样本按照大小排序，取中间位置的元素。

**若样本数量为奇数，中位数为最中间的元素**

$[1, 2000, 3000, 4000, 10000000]$

**若样本数量为偶数，中位数为最中间的两个元素的平均值**

$[1,2000,3000,4000,5000,10000000]$

案例：分析中位数的算法，测试numpy提供的中位数API：

```python
import numpy as np
closing_prices = np.loadtxt( '../../data/aapl.csv', 
	delimiter=',', usecols=(6), unpack=True)
size = closing_prices.size
sorted_prices = np.msort(closing_prices)
median = (sorted_prices[int((size - 1) / 2)] + sorted_prices[int(size / 2)]) / 2
print(median)
median = np.median(closing_prices)
print(median)
```

### 标准差

样本：$S = [s_1, s_2, s_3, ..., s_n]$

平均值：$m = \frac{s_1 + s_2 + s_3 + ... + s_n}{n}$

离差：$D = [d_1, d_2, d_3, ..., d_n]; d_i = S_i-m$

离差方：$Q = [q_1, q_2, q_3, ..., q_n]; q_i=d_i^2$

总体方差：$v = \frac{(q_1+q_2+q_3 + ... + q_n)}{n}$

总体标准差：$s = \sqrt{v}$

样本方差：$v' = \frac{(q_1+q_2+q_3 + ... + q_n)}{n-1}$

样本标准差：$s' = \sqrt{v'}$

```python
import numpy as np
closing_prices = np.loadtxt(
    '../../data/aapl.csv', delimiter=',', usecols=(6), unpack=True)
mean = np.mean(closing_prices)         # 算数平均值
devs = closing_prices - mean           # 离差
dsqs = devs ** 2                       # 离差方
pvar = np.sum(dsqs) / dsqs.size        # 总体方差
pstd = np.sqrt(pvar)                   # 总体标准差
svar = np.sum(dsqs) / (dsqs.size - 1)  # 样本方差
sstd = np.sqrt(svar)                   # 样本标准差
print(pstd, sstd)
pstd = np.std(closing_prices)          # 总体标准差
sstd = np.std(closing_prices, ddof=1)  # 样本标准差
print(pstd, sstd)
```

### 数组的轴向汇总

案例：统计每个周一，周二。。。周五的收盘价的平均值，并放入一个数组

```python
import datetime as dt
import numpy as np

# 转换器函数： 将日-月-年格式的日期字符串转换为星期
def dmy2wday(dmy):
  dmy = str(dmy, encoding = 'uft-8')
  date = dt.datetime.striptime(dmy, '%d-%m-%y').date
  wday = date.weekday() #用 周日
  return wday

wdays, closing_prices = np.loadtxt('../data/aapl.csv', delimiter=',', usecols=(1, 6), unpack=True, converters={1:dmy2wday})

print(wdays == 0)
```

```py
# 案例
avg_prices = np.zeros(5)
for i in range(avg_prices.size):
  avg_prices[i] = np.mean(closing_prices[wdays == i])
  
print(avg_prices)

```



### 数组的轴向汇总

案例：汇总每周的最高价，最低价，开盘价，收盘价。

```python
def func(data):
    pass
#func 	处理函数
#axis 	轴向 [0,1]
#array 	数组
np.apply_along_axis(func, axis, array)

```

沿着数组中所指定的轴向，调用处理函数，并将每次调用的返回值重新组织成数组返回。

```python
wdays, opening_prices, highest_prices, \
    lowest_prices, closing_prices = np.loadtxt(
        '../data/aapl.csv',
        delimiter=',', usecols=(1, 3, 4, 5, 6),
        unpack=True, converters={1: dmy2wday})

first_mon = np.where(wdays==0)[0][0]
last_fri = np.where(wdays==4)[0][-1]

wdays = wdays[first_mon:last_fri+1]
indices = np.arange(first_mon, last_fri+1)

#把周一至周五每天的indices值统计为5个数组
mon_indices = indices[wdays==0]
tue_indices = indices[wdays==1]
wen_indices = indices[wdays==2]
thu_indices = indices[wdays==3]
fri_indices = indices[wdays==4]
max_len = np.max((mon_indices.size, tue_indices.size, wen_indices.size, thu_indices.size, fri_indices.size))
mon_indices = np.pad(mon_indices, pad_width=(0, max_len-mon_indices.size), mode='constant', constant_values=-1)
indices = np.vstack((mon_indices,tue_indices,wen_indices,thu_indices,fri_indices))

# numpy将会把每一行的indices传入summary函数执行业务
def summary(indices):
    indices = indices[indices!=-1]
    opening_price = opening_prices[indices[0]]
    highest_price = highest_prices[indices].max()
    lowest_price = lowest_prices[indices].min()
    closing_price = closing_prices[indices[-1]]
    return opening_price, highest_price, lowest_price, closing_price
	
r = np.apply_along_axis(summary, 1, indices)
print(r)

np.savetxt('../../data/summary.csv', summaries, delimiter=',', fmt='%g')


```

### 移动均线

收盘价5日均线：从第五天开始，每天计算最近五天的收盘价的平均值所构成的一条线。

移动均线算法：

```python
(a+b+c+d+e)/5
(b+c+d+e+f)/5
(c+d+e+f+g)/5
...
(f+g+h+i+j)/5

```

在K线图中绘制5日均线图

```python
import datetime as dt
import numpy as np
import matplotlib.pyplot as mp
import matplotlib.dates as md

def dmy2ymd(dmy):
    dmy = str(dmy, encoding='utf-8')
    date = dt.datetime.strptime(dmy, '%d-%m-%Y').date()
    ymd = date.strftime('%Y-%m-%d')
    return ymd

dates, closing_prices = np.loadtxt('../data/aapl.csv', delimiter=',',
    usecols=(1, 6), unpack=True, dtype='M8[D], f8', converters={1: dmy2ymd})
sma51 = np.zeros(closing_prices.size - 4)
for i in range(sma51.size):
    sma51[i] = closing_prices[i:i + 5].mean()
# 开始绘制5日均线
mp.figure('Simple Moving Average', facecolor='lightgray')
mp.title('Simple Moving Average', fontsize=20)
mp.xlabel('Date', fontsize=14)
mp.ylabel('Price', fontsize=14)
ax = mp.gca()
# 设置水平坐标每个星期一为主刻度
ax.xaxis.set_major_locator(md.WeekdayLocator( byweekday=md.MO))
# 设置水平坐标每一天为次刻度
ax.xaxis.set_minor_locator(md.DayLocator())
# 设置水平坐标主刻度标签格式
ax.xaxis.set_major_formatter(md.DateFormatter('%d %b %Y'))
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
dates = dates.astype(md.datetime.datetime)
mp.plot(dates, closing_prices, c='lightgray', label='Closing Price')
mp.plot(dates[4:], sma51, c='orangered', label='SMA-5(1)')
mp.legend()
mp.gcf().autofmt_xdate()
mp.show()

```

#### 卷积

先理解卷积运算的过程：

```python
a = [1, 2, 3, 4, 5]   源数组
b = [8, 7, 6]		  卷积核  kernel
使用b作为卷积核，对a数组执行卷积运算的过程如下：

			   44  65  86           有效卷积 (valid)
           23  44  65  86  59       同维卷积 (same)
        8  23  44  65  86  59  30   完全卷积 (full)
0   0   1   2   3   4   5   0   0
6   7   8
	6   7   8
    	6   7   8
        	6   7   8
            	6   7   8
                	6   7   8
                    	6   7   8

c = numpy.convolve(a, b, 卷积类型)

```

**5日移动均线序列可以直接使用卷积实现**

```python
a = [a, b, c, d, e, f, g, h, i, j] 
b = [1/5, 1/5, 1/5, 1/5, 1/5]

```

**使用卷积函数numpy.convolve(a, b, 卷积类型)实现5日均线**

```python
sma52 = np.convolve( closing_prices, np.ones(5) / 5, 'valid')
mp.plot(dates[4:], sma52, c='limegreen', alpha=0.5,
        linewidth=6, label='SMA-5(2)')

```

**使用卷积函数numpy.convolve(a, b, 卷积类型)实现10日均线**

```python
sma10 = np.convolve(closing_prices, np.ones(10) / 10, 'valid')
mp.plot(dates[9:], sma10, c='dodgerblue', label='SMA-10')

```

**使用卷积函数numpy.convolve(a, b, 卷积类型)实现加权5日均线**

```python
weights = np.exp(np.linspace(-1, 0, 5))
weights /= weights.sum()
ema5 = np.convolve(closing_prices, weights[::-1], 'valid')
mp.plot(dates[4:], sma52, c='limegreen', alpha=0.5,
        linewidth=6, label='SMA-5')
```

### 布林带

布林带由三条线组成：

中轨：移动平均线

上轨：中轨+2x5日收盘价标准差	（顶部的压力）

下轨：中轨-2x5日收盘价标准差 	（底部的支撑力）

布林带收窄代表稳定的趋势，布林带张开代表有较大的波动空间的趋势。

**绘制5日均线的布林带**

```python
weights = np.exp(np.linspace(-1, 0, 5))
weights /= weights.sum()
em5 = np.convolve(closing_prices, weights[::-1], 'valid')
stds = np.zeros(em5.size)
for i in range(stds.size):
    stds[i] = closing_prices[i:i + 5].std()
stds *= 2
lowers = medios - stds
uppers = medios + stds

mp.plot(dates, closing_prices, c='lightgray', label='Closing Price')
mp.plot(dates[4:], medios, c='dodgerblue', label='Medio')
mp.plot(dates[4:], lowers, c='limegreen', label='Lower')
mp.plot(dates[4:], uppers, c='orangered', label='Upper')
```

## 











