from flask import Flask, render_template, request
import pandas as pd
import implicit
from scipy.sparse import csr_matrix

app = Flask(__name__)

# Placeholder data for initial model training
df = pd.DataFrame({
    'Customerid': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Productid': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P009', 'P010'],
    'Rating': [4.5, 3.2, 4.8, 3.9, 2.7, 4, 3.5, 4.2, 2.5, 3.8]
})

# Convert the user-item matrix to a sparse matrix
user_item_matrix = pd.pivot_table(df, values='Rating', index='Customerid', columns='Productid', fill_value=0)
sparse_user_item = csr_matrix(user_item_matrix.values)

# Build the ALS model
model = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.01, iterations=20)
model.fit(sparse_user_item)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Check if the post request has the file part
            if 'file' not in request.files:
                return render_template('index.html', error='No file part')

            file = request.files['file']

            # If the user does not select a file, browser also
            # submit an empty part without filename
            if file.filename == '':
                return render_template('index.html', error='No selected file')

            # Read the uploaded CSV file
            uploaded_df = pd.read_csv(file)

            # Convert 'Rating' column to numeric type
            uploaded_df['Rating'] = pd.to_numeric(uploaded_df['Rating'], errors='coerce')

            # Assuming the CSV file has columns 'Customerid' and 'Rating'
            user_id = uploaded_df['Customerid'].iloc[0]

            # Create a user-item interaction matrix for the uploaded data
            uploaded_user_item_matrix = pd.pivot_table(uploaded_df, values='Rating', index='Customerid', columns='Productid', fill_value=0)
            uploaded_sparse_user_item = csr_matrix(uploaded_user_item_matrix.values)

            # Recommend products for the specified user in the uploaded data
            recommended_items = model.recommend(user_id, uploaded_sparse_user_item, N=5)

            # Extract product IDs and scores
            recommendations = [{'Productid': item_id, 'Score': score} for item_id, score in recommended_items]

            return render_template('index.html', user_id=user_id, recommendations=recommendations)
        except Exception as e:
            return render_template('index.html', error=f'Error: {e}')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
