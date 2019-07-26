import flask
from flask import Flask, request, render_template, jsonify, Markup

app = Flask(__name__)


def process_query(query):
    return "Hello there .. " + query


def http_query(query):
    query_init = query
    query = query.strip().lower()

    if query == 'show example':
        result = gpopu.query_examples()
        response = "<h2>Response:</h2><h3 style='font-family:courier;color:navy'>%s</h3>" % result
        response = "<h2 style='color:green'>Query: <div style=color:#FF4500> {}!</div></h2>".format(query_init) + response
        return response

    if query.find('show dim') > -1 or query.find('show context') > -1 or query.find('show axis') > -1:
        result = gpopu.context_dimensions()
        response = "<h2>Response:</h2><h3 style='font-family:courier;color:navy'>%s</h3>" % result
        response = "<h2 style='color:green'>Query: <div style=color:#FF4500> {}!</div></h2>".format(query_init) + response
        return response

    if query.find('show total revenue') > -1:
        result = gpopu.query_eg01()
        response = "<h2>Response:</h2><h3 style='font-family:courier;color:navy'>%s</h3>" % result
        response = "<h2 style='color:green'>Query: <div style=color:#FF4500> {}!</div></h2>".format(query_init) + response
        return response

    if query.find('show context total revenue') > -1:
        result = gpopu.query_category_02('Total Revenue')
        response = "<h2>Response:</h2><h3 style='font-family:courier;color:navy'>%s</h3>" % result
        response = "<h2 style='color:green'>Query: <div style=color:#FF4500> {}!</div></h2>".format(query_init) + response
        return response

    if query.find('show fte headcount') > -1:
        result = gpopu.query_eg02()
        response = "<h2>Response:</h2><h3 style='font-family:courier;color:navy'>%s</h3>" % result
        response = "<h2 style='color:green'>Query: <div style=color:#FF4500> {}!</div></h2>".format(query_init) + response
        return response

    if query.find('show contractors headcount') > -1:
        result = gpopu.query_eg02b()
        response = "<h2>Response:</h2><h3 style='font-family:courier;color:navy'>%s</h3>" % result
        response = "<h2 style='color:green'>Query: <div style=color:#FF4500> {}!</div></h2>".format(query_init) + response
        return response

    if query.find('show non fte headcount') > -1:
        result = gpopu.query_eg03()
        response = "<h2>Response:</h2><h3 style='font-family:courier;color:navy'>%s</h3>" % result
        response = "<h2 style='color:green'>Query: <div style=color:#FF4500> {}!</div></h2>".format(query_init) + response
        return response

    if query.find('show stats for relations') > -1 or query.find('show relations') > -1 \
            or query.find('show me relations') > -1:
        result = gpopu.query_eg04()
        response = "<h2>Response:</h2><h3 style='font-family:courier;color:navy'>%s</h3>" % result
        response = "<h2 style='color:green'>Query: <div style=color:#FF4500> {}!</div></h2>".format(query_init) + response
        return response

    if query.find('show stats for classes') > -1 or query.find('show classes') > -1 or query.find('show me classes') > -1:
        result = gpopu.query_eg05()
        response = "<h2>Response:</h2><h3 style='font-family:courrier;color:navy'>%s</h3>" % result
        response = "<h2 style='color:green'>Query: <div style=color:#FF4500> {}!</div></h2>".format(query_init) + response
        return response

    if query.find('show vocab') > -1 or query.find('show me vocab') > -1:
        result = gpopu.query_eg07()
        response = "<h2>Response:</h2><h3 style='font-family:courrier;color:navy'>%s</h3>" % result
        response = "<h2 style='color:green'>Query: <div style=color:#FF4500> {}!</div></h2>".format(query_init) + response
        return response

    if query.find('help') > -1 or query.find('?') > -1:
        help = """
            To try KGIS enter this query: <b>show example</b><br>
            These are search phrases (not parsed):
            <ul>
            <li>help or ?</li>
            <li>parse expression will return the expression parsed JSON</li>
            <UL><li> e.g. parse show me revenue for FY19 range min 1000 max 2000</li></ul>
            <li>Filter Syntax: query {range min number | range max number | range min number max number</li>
            <li>show dim</li>
            <li>show axis</li>
            <li>show context</li>
            <li>show total revenue</li>
            <li>show context total revenue</li>
            <li>show fte headcount</li>
            <li>show non fte headcount</li>
            <li>show contractors headcount</li>
            <li>show relations</li>
            <li>show stats for relations</li>
            <li>show classes</li>
            <li>show stats for classes</li>
            <li>show vocabulary</li>
            </ul>
            Parsed search:
            <ul>
            <li>context : show me revenue for FY19</li>
            <li>FY19_AOP: show me revenue</li>
            <li>context : find me revenue for last month range min number max number</li>
            <li>show me dashboard for "OUT04 : P&L by Nature"</li>
            <li>give me full time employees headcount</li>
            <li>give me non full time employees headcount</li>
            <li>show me number of employee</li>
            <li>show number of employees quarterly</li>
            <li>show number of employees monthly</li>
            <li>how many employees</li>
            <li>give me contractors headcount</li>
            <li>show formulas</li>
            <li>show formulas for revenue</li>
            <li>show formulas contain revenue</li>
            <li>show me formula for "Service & Other Revenue"</li>
            <li>show total number of employee last month</li>
            <li>show me dashboard for “OUT03 : P&L by Nature Headcount”</li>
            <li>show me all the new hires in September 2018</li>
            <li>show me all the new hires for 2018-09-24</li>
            <li>Show me what REQ will backfill Nicol-Perry, Yelena</li>
            </ul>
            <br><br>
            
        """
        return help


@app.route('/')
def homepage():
    return render_template('index.html')


@app.route('/', methods=["POST"])
def deal_with_query():
    query = request.form.get('story')
    results = http_query(query)
    print(query)
    return render_template('index.html', results=Markup(results), query=query)
    # return ('', 204)

# @app.route('/receive_data')
# def get_id():
#     the_id = flask.request.args.get('text_input')
#     return "<p>Got it!</p>"


if __name__ == "__main__":
    # app.debug=True
    app.run(host='0.0.0.0', threaded=True)
