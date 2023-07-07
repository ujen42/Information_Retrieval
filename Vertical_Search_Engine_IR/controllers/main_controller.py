from flask.views import MethodView
from flask import render_template, redirect, request
from utils.search import Search
from utils.classifier import NaiveBayesClassifierUtil

class Home(MethodView):
    def get(self):
        return render_template("main/index.html")
    
    def post(self):
        form_data = request.form
        query = form_data.get('query')
        return redirect(f"/result?query={query}")


class ResultPage(MethodView):
    def get(self):
        query = request.args.get('query')
        if(query is None): return redirect('/')
        search = Search(query)
        result = search.search()
        return render_template("main/resultpage.html", query=query, result = result)
    
class NewsSubjectClassify(MethodView):
    
    def get(self):
         return render_template("main/classifier.html")
    def post(self):
        print("----------------")
        form_data = request.form
        query = form_data.get('news')
        obj = NaiveBayesClassifierUtil()
        result = obj.nb_classify(query)
        print("hello world")
        print(result)
        return render_template("main/classifier.html", result = result)
    

