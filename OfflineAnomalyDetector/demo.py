from Dispatchers import Dispatcher
from Fetchers import CsvFetcher
from sklearn.preprocessing import MinMaxScaler
from Detectors import ExpAvgD, ShesdD, LstmAutoEncoder#, LstmAutoEncoder
import ray, time
import numpy as np

def main():
	ray.init()
	dt = ShesdD()
	a = time.time()
	df = CsvFetcher("../../data/all")
	a2 = Dispatcher(dt,df,4)
	a2.run()
	print time.time() - a

def main2():
    df = CsvFetcher("../data")
    model_dir_path = './model'
    #data = pd.read_csv("../data/"+df.getData()['name'], header=None).as_matrix()[1:1000,1]
    frame = df.getData(5)
    data = np.array([x['data'] for x in frame])
    print frame[0]['name']
    print("data.head()")
    print(data)
    print("data.head()")
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    ae = LstmAutoEncoder()
    ae.fit(data, model_dir_path, time_window_size = 50, epochs = 5)
    ae.load_model(model_dir_path)
    print ae.predict(data)

if __name__ == '__main__':
	main()