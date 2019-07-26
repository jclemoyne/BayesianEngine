from flask import Flask
from flask import render_template, send_from_directory
import pandas as pd

# app = Flask(__name__, root_path='jQueryTableEdit/')
# app = Flask(__name__, root_path='TableEdit_eg/')
# app = Flask(__name__, static_url_path='/Users/jclaudel/flaskstatic/')
# app = Flask(__name__)
# app = Flask(__name__, root_path='/Users/jclaudel/flaskstatic/')
static_folder = '/Users/jclaudel/work/Data/Bayes/BayesianEngine/KnowledgeMachine/TableEdit_eg/static'
app = Flask(__name__, static_url_path="/static", static_folder=static_folder)


def read_excel(xclfile='data/Financial_Sample.xlsx'):
	df = pd.read_excel(xclfile)
	# print(df.head())
	return df


def render_html(df):
	# with open('css/df_style.css', 'r') as cssfile:
	# 	style = cssfile.read()
	jqscript = """<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>"""
	docready = """<script>
	$( document ).ready(function() {
    $('table td').last().attr("contenteditable", true);});
    </script>"""
	tdscript = """<script>$('table td').last().attr("contenteditable", true);</script>"""
	style = """
<style>
.mystyle {
    font-size: 11pt;
    font-family: Arial;
    border-collapse: collapse;
    border: 1px solid silver;

}

.mystyle td, th {
    padding: 5px;
}

.mystyle tr:nth-child(even) {
    background: #E0E0E0;
}

.mystyle tr:hover {
    background: silver;
    cursor: pointer;
}
</style>	
	"""
	html_str = """<html><head><head><title>HTML Pandas Dataframe with CSS</title></head></head>
	{2}
	{3}
	{1}<body><div>{0}</div></body></html>"""
	return html_str.format(df.to_html(classes="mystyle"), style, jqscript, docready)


@app.route("/")
def hello():
	# return "<h1>Hello Sheets"
	df = read_excel()
	return render_html(df)


@app.route("/edit")
def editable():
	# return render_template('index.html', message='Hello')
	# print('===  config: ', app.config['UPLOAD_FOLDER'])
	# return send_from_directory(directory='TableEdit_eg/static/', filename='index.html')
	return app.send_static_file('index.html')


if __name__ == '__main__':
	read_excel()
	app.run(host="0.0.0.0", threaded=True)