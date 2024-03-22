import pandas as pd
import requests

# Function to check if a URL is valid
def is_valid_url(url):
    try:
        response = requests.head(url)
        return response.status_code == 200
    except Exception as e:
        print(f"Error checking URL {url}: {str(e)}")
        return False

# Read the CSV file, skipping the first row
df = pd.read_csv("A2_Data.csv", skiprows=1, header=None, names=["ID", "Image", "Review Text"])

# Initialize an empty list to store the new rows
new_rows = []
invalid_urls = []

# Iterate over each row in the dataframe
for index, row in df.iterrows():
    # Split the URLs in the "Image" column
    image_urls = row["Image"].strip('[]').replace("'", "").split(", ")
    
    # Check if each URL is valid
    valid_urls = []
    for url in image_urls:
        if is_valid_url(url):
            valid_urls.append(url)
        else:
            invalid_urls.append(url)
    
    # Create a new row for each valid URL
    for url in valid_urls:
        new_rows.append({"ID": row["ID"], "Image": url, "Review Text": row["Review Text"]})

# Print the names of invalid URLs
print("Invalid URLs:")
for url in invalid_urls:
    print(url)

# Create a new dataframe from the new rows
new_df = pd.DataFrame(new_rows)

# Save the new dataframe to a new CSV file
new_df.to_csv("A2_Data_modified.csv", index=False)