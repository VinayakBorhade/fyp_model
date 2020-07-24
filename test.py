import predict_csic as model
import predict_nsl_kdd as model2
import predict_ecml_pkdd as model3
print("----------------calling the csic model----------------")
X = model._readLogs(file_path="xss.txt")
y = model._predict(X)

print("----------------calling nsl kdd model----------------")
X = model2._readLogs(file_path="KDDTest.txt")
y = model2._predict(X)
print(y)

print("----------------calling ecml pkdd model----------------")
X = model3._readLogs(file_path="tomcat_attack.txt")
y = model3._predict(X)
print(y)