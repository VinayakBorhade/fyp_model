import predict as model
print("calling the model")
X = model._readLogs(file_path="xss.txt")
y = model._predict(X)