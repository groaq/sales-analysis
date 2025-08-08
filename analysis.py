import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Loading the dataset.
def load_sales_data(filepath):
    return pd.read_csv(filepath, encoding='latin1')

# Cleans the sales data by removing duplicates, converting dates, and standardizing text columns.
def clean_sales_data(df):
    df = df.copy()

    # Drop duplicates.
    df.drop_duplicates(inplace=True)

    # Convert date columns.
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce')

    # Convert Postal Code to string.
    df['Postal Code'] = df['Postal Code'].astype(str)

    # Strip whitespace from text columns.
    text_cols = ['Customer Name', 'Segment', 'Country', 'City', 'State',
                 'Region', 'Category', 'Sub-Category', 'Product Name']
    for col in text_cols:
        df[col] = df[col].str.strip()

    # Drop rows with missing key dates or sales info.
    df.dropna(subset=['Order Date', 'Ship Date', 'Sales'], inplace=True)

    return df

# Identifies data issues such as negative sales or invalid discounts.
def validate_columns(df):
    issues = {
        'negative_sales': df[df['Sales'] < 0],
        'invalid_quantity': df[df['Quantity'] <= 0],
        'invalid_discount': df[(df['Discount'] < 0) | (df['Discount'] > 1)],
        'extreme_profit': df[(df['Profit'] < -10000) | (df['Profit'] > 10000)],
    }
    return issues

# Prints a summary of data validation issues.
def report_validation_issues(df):
    validation_results = validate_columns(df)

    for issue, rows in validation_results.items():
        print(f"\n{issue} — {len(rows)} rows")
        if not rows.empty:
            display(rows.head())

# Summarizes overall sales performance metrics including totals and averages.
def sales_performance(df):
    # Creating a dictionary for the summary.
    summary = {
        'total_sales': df['Sales'].sum(),
        'total_profit': df['Profit'].sum(),
        'total_orders': df['Order ID'].nunique(),
        'average_discount': df['Discount'].mean(),
        'most_common_category': df['Category'].mode()[0],
        'most_common_region': df['Region'].mode()[0]
    }
    # Creating a data frame of the summary.
    summary_df = pd.DataFrame.from_dict(summary, orient='index', columns=['Value'])
    
    # Format specific rows.
    summary_df.loc['total_sales', 'Value'] = f"${summary['total_sales']:,.2f}"
    summary_df.loc['total_profit', 'Value'] = f"${summary['total_profit']:,.2f}"
    summary_df.loc['average_discount', 'Value'] = f"{summary['average_discount']:.2%}"

    return summary_df

# Returns the top 10 products ranked by sales, with formatted dollar amounts.
def top_sales_products(df):
    top_products = df.groupby('Product Name').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
    top_products = top_products.sort_values(by='Sales', ascending=False).head(10).reset_index(drop=True)
    
    # Add a rank column from 1 to 10.
    top_products.index = top_products.index + 1
    top_products.index.name = 'Rank'

    # Format Profit and Sales as dollar amounts.
    top_products['Sales'] = top_products['Sales'].apply(lambda x: f"${x:,.2f}")
    top_products['Profit'] = top_products['Profit'].apply(lambda x: f"${x:,.2f}")
    
    return top_products

# Returns the top 10 products ranked by profit, with formatted dollar amounts.
def top_profit_products(df):
    top_products = df.groupby('Product Name').agg({'Profit': 'sum', 'Sales': 'sum'}).reset_index()
    top_products = top_products.sort_values(by='Profit', ascending=False).head(10).reset_index(drop=True)
    
    # Add a rank column from 1 to 10.
    top_products.index = top_products.index + 1
    top_products.index.name = 'Rank'

    # Format Profit and Sales as dollar amounts.
    top_products['Profit'] = top_products['Profit'].apply(lambda x: f"${x:,.2f}")
    top_products['Sales'] = top_products['Sales'].apply(lambda x: f"${x:,.2f}")
    
    return top_products

# Summarizes total profit by category, sorted and ranked.
def profit_per_category(df):
    profit_by_category = df.groupby('Category').agg({'Profit': 'sum'}).reset_index()
    profit_by_category = profit_by_category.sort_values(by='Profit', ascending=False).reset_index(drop=True)
    
    # Add a rank column from 1 to N.
    profit_by_category.index = profit_by_category.index + 1
    profit_by_category.index.name = 'Rank'

    # Format Profit as dollar amounts.
    profit_by_category['Profit'] = profit_by_category['Profit'].apply(lambda x: f"${x:,.2f}")
    
    return profit_by_category

# Summarizes total profit by sub-category, sorted and ranked.
def profit_per_subcategory(df):
    profit_by_subcategory = df.groupby('Sub-Category').agg({'Profit': 'sum'}).reset_index()
    profit_by_subcategory = profit_by_subcategory.sort_values(by='Profit', ascending=False).reset_index(drop=True)
    
    # Add a rank column from 1 to N.
    profit_by_subcategory.index = profit_by_subcategory.index + 1
    profit_by_subcategory.index.name = 'Rank'

    # Format Profit as dollar amounts. 
    profit_by_subcategory['Profit'] = profit_by_subcategory['Profit'].apply(lambda x: f"${x:,.2f}")
    
    return profit_by_subcategory

# Aggregates total sales by year with formatted values.
def sales_over_years(df, time_col='Order Date', agg_col='Sales'):
    df['Year'] = df[time_col].dt.year
    trends = df.groupby('Year').agg({agg_col: 'sum'}).reset_index()
    trends.rename(columns={agg_col: 'Total Sales'}, inplace=True)
    
    # Format Total Sales as dollar amounts. 
    trends['Total Sales'] = trends['Total Sales'].apply(lambda x: f"${x:,.2f}")
    
    return trends

# Aggregates total sales by month with formatted values.
def sales_over_months(df, time_col='Order Date', agg_col='Sales'):
    # Extract full month name.
    df['Month'] = df[time_col].dt.strftime('%B')

    # To ensure months appear in calendar order.
    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)

    trends = df.groupby('Month').agg({agg_col: 'sum'}).reset_index()
    trends.rename(columns={agg_col: 'Total Sales'}, inplace=True)

    # Format Total Sales as dollar amounts.
    trends['Total Sales'] = trends['Total Sales'].apply(lambda x: f"${x:,.2f}")

    return trends.sort_values('Month')

# Provides sales and profit aggregated by state, sorted and formatted.
def geographic_insights(df):
    # Group by Country and State, aggregating Sales and Profit.
    geo_insights = df.groupby('State').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
    
    # Sort by Sales in descending order
    geo_insights = geo_insights.sort_values(by='Sales', ascending=False).reset_index(drop=True)
    
    # Format Sales and Profit as dollar amounts.
    geo_insights['Sales'] = geo_insights['Sales'].apply(lambda x: f"${x:,.2f}")
    geo_insights['Profit'] = geo_insights['Profit'].apply(lambda x: f"${x:,.2f}")
    
    return geo_insights

# Summarizes sales and profit by customer segment, sorted and formatted.
def segment_analysis(df):
    # Group by Segment and aggregate Sales and Profit.
    segment_summary = df.groupby('Segment').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
    
    # Sort by Sales in descending order.
    segment_summary = segment_summary.sort_values(by='Sales', ascending=False).reset_index(drop=True)
    
    # Format Sales and Profit as dollar amounts.
    segment_summary['Sales'] = segment_summary['Sales'].apply(lambda x: f"${x:,.2f}")
    segment_summary['Profit'] = segment_summary['Profit'].apply(lambda x: f"${x:,.2f}")
    
    return segment_summary

# Calculates average, minimum, and maximum order-to-ship times in days.
def order_to_ship_summary(df):
    # Subtract the difference between Ship Date and Order Date.
    shipping_times = df['Ship Date'] - df['Order Date']

    # Extract the days from the calculations and store in a dictionary.
    summary = {
        'order_to_ship_average' : shipping_times.dt.days.mean(),
        'order_to_ship_min' : shipping_times.dt.days.min(),
        'order_to_ship_max' : shipping_times.dt.days.max()
    }

    # Return as a DataFrame.
    return pd.DataFrame.from_dict(summary, orient='index', columns=['Days'])
    

# Summarizes sales, profit, and quantity grouped by discount ranges.
def get_discount_impact_summary(df):
    df = df.copy()
    df['Discount Bin'] = pd.cut(df['Discount'], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0])
    summary = df.groupby('Discount Bin').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    
    return summary

# Summarizes average discount and discounted order counts by category and sub-category.
def category_discount_summary(df):
    category_discount = df[df['Discount'] > 0].groupby(['Category', 'Sub-Category']).agg({
    'Discount': ['mean', 'count']}).sort_values(('Discount', 'mean'), ascending=False)
    category_discount.columns = ['Avg Discount', 'Discounted Orders']
    category_discount.reset_index(inplace=True)
    
    return category_discount

# Plots Discount vs. Profit scatterplot and returns their correlation matrix.
def plot_profit_vs_discount(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Discount', y='Profit')
    plt.title("Discount vs. Profit")
    plt.show()
    
    # Also check correlation.
    return df[['Discount', 'Profit']].corr()

# Plots monthly sales trend line chart.
def plot_monthly_sales_trend(df):
    df['Order Month'] = pd.to_datetime(df['Order Date']).dt.to_period('M')
    monthly_sales = df.groupby('Order Month')['Sales'].sum()

    plt.figure(figsize=(12,6))
    monthly_sales.plot(marker='o')
    plt.title('Monthly Sales Trend')
    plt.xlabel('Month')
    plt.ylabel('Total Sales')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plots a horizontal bar chart of the top N products by sales.
def plot_top_products_by_sales(df, top_n=10):
    top_products = df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10,6))
    top_products.plot(kind='barh')
    plt.title(f'Top {top_n} Products by Sales')
    plt.xlabel('Total Sales')
    plt.ylabel('Product Name')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


