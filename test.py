import predict_csic as model
import predict_nsl_kdd as model2
print("----------------calling the csic model----------------")
X = model._readLogs(file_path="xss.txt")
y = model._predict(X)

print("----------------calling nsl kdd model----------------")
X = model2._readLogs(file_path="slowloris.csv")
y = model2._predict(X)
print(y)