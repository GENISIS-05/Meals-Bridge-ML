from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

df = pd.read_csv('./slums_data.csv')

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET', 'POST'])
def distribute_to_api():
    try:
        waste = int(request.args.get('waste'))
        k_val = int(request.args.get('k'))
        # desc = df['Total'].describe()

        # if waste < desc['25%']:
        #     res = df[df['Category'] == 'Low']['Location'].tolist()
        # elif waste < desc['75%']:
        #     res = df[df['Category'] == 'Medium']['Location'].tolist()
        # else:
        #     res = df[df['Category'] == 'High']['Location'].tolist()

        # return str(res)
        def find_min_differences(target_number, values):
            differences = [(value, abs(target_number - value)) for value in values]
            sorted_differences = sorted(differences, key=lambda x: x[1])
            min_diff_values = [item[0] for item in sorted_differences[:k_val]]
            return min_diff_values
        mins = find_min_differences(waste,df['Total'])
        res = df[df['Total'].isin(mins)]['Location'].values
        return list(res[:k_val])

    except Exception as e:
        error_message = str(e)
        return 'Error'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
