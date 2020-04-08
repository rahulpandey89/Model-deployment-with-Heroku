{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = {'exeperince':['five','two','seven','three','ten','eleven'],'test_score':['8','8','6','10','9','7']\n",
    "                      ,'interview_score':['9','6','7','10','6','10'],'salary':['50000','45000','60000','65000','70000','62000']}\n",
    "\n",
    "#dataset['experience'].fillna(0, inplace=True)\n",
    "\n",
    "#dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)\n",
    "\n",
    "#X = dataset.iloc[:, :3]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'exeperince': ['five', 'two', 'seven', 'three', 'ten', 'eleven'],\n",
       " 'test_score': ['8', '8', '6', '10', '9', '7'],\n",
       " 'interview_score': ['9', '6', '7', '10', '6', '10'],\n",
       " 'salary': ['50000', '45000', '60000', '65000', '70000', '62000']}"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset=pd.DataFrame(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>exeperince</th>\n",
       "      <th>test_score</th>\n",
       "      <th>interview_score</th>\n",
       "      <th>salary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>five</td>\n",
       "      <td>8</td>\n",
       "      <td>9</td>\n",
       "      <td>50000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>two</td>\n",
       "      <td>8</td>\n",
       "      <td>6</td>\n",
       "      <td>45000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>seven</td>\n",
       "      <td>6</td>\n",
       "      <td>7</td>\n",
       "      <td>60000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>three</td>\n",
       "      <td>10</td>\n",
       "      <td>10</td>\n",
       "      <td>65000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ten</td>\n",
       "      <td>9</td>\n",
       "      <td>6</td>\n",
       "      <td>70000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>eleven</td>\n",
       "      <td>7</td>\n",
       "      <td>10</td>\n",
       "      <td>62000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  exeperince test_score interview_score salary\n",
       "0       five          8               9  50000\n",
       "1        two          8               6  45000\n",
       "2      seven          6               7  60000\n",
       "3      three         10              10  65000\n",
       "4        ten          9               6  70000\n",
       "5     eleven          7              10  62000"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = dataset.iloc[:, :3]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\admin\\AppData\\Local\\Continuum\\anaconda3\\lib\\site-packages\\ipykernel_launcher.py:6: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  \n"
     ]
    }
   ],
   "source": [
    "def convert_to_int(word):\n",
    "    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,\n",
    "                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}\n",
    "    return word_dict[word]\n",
    "\n",
    "X['exeperince'] = X['exeperince'].apply(lambda x : convert_to_int(x))\n",
    "\n",
    "y = dataset.iloc[:, -1]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[69663.89987679]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "regressor = LinearRegression()\n",
    "\n",
    "#Fitting model with trainig data\n",
    "regressor.fit(X, y)\n",
    "\n",
    "# Saving model to disk\n",
    "pickle.dump(regressor, open('model.pkl','wb'))\n",
    "\n",
    "# Loading model to compare the results\n",
    "model = pickle.load(open('model.pkl','rb'))\n",
    "print(model.predict([[10, 9, 6]]))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
