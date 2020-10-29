# import pandas as pd
# def load_data():
#         """Reads the data from the csv files , removes the rows with missing value and resets the index"""
#         data=pd.read_csv('src/pages/Home/Dept.csv')
#         data['I'] = pd.to_numeric(data['I'].str.replace('Nil', ''))
#         data['II'] = pd.to_numeric(data['II'].str.replace('Nil', ''))
#         data['III'] = pd.to_numeric(data['III'].str.replace('Nil', ''))
#         data = data.dropna()
#         data = data.reset_index(drop=True)
#         return data
def attendance(arr,dat):
        for i in range(len(arr)+1):
                for j in dat:
                        if i+1 == j:
                                arr[i] -= 0.5
                                if arr[i] == -0.5:
                                        arr[i] = 0
                                dat.remove(i+1)
        return '\n'.join(arr)

arr =[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
dat= [27,5,15,27,41,5,9,27]
print(attendance(arr,dat))

1
2
