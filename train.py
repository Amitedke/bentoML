import bentoml
from sklearn import svm
from sklearn import datasets

iris = datasets.load_iris()
X,y = iris.data,iris.target


clf = svm.SVC(gamma='scale')
clf.fit(X,y)

saved_model = bentoml.sklearn.save_model("iris_clf",clf)

print(f"model saved : {saved_model}")


#tag="iris_clf:eyvflfk3xgpbaldn"
#tag="iris_clf:ioa4o2k3xgn7eldn"

#bentoml models list ---> cmd bentoml models list