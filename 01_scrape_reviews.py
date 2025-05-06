# import required libraries
import pandas as pd
from google_play_scraper import app, Sort, reviews
from pprint import pprint
import pymongo
import datetime as dt
from tzlocal import get_localzone
import random
import time

client = pymongo.MongoClient(
    "mongodb+srv://rishikavaish321:SQEfC5ZCpmwrwg8F@cluster0.qzsdybi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)
app_proj_db = client['playstore_reviews']  # Using a consistent DB name
review_collection = app_proj_db['bank_reviews']  # Collection where reviews are stored

# choosing some banking app names along with their app IDs
app_names = [
    'SBI',
'UNION',
'CBOI',
'BOI',
]

app_ids =  [
  'com.freedomrewardz',
'com.unionrewardz',
'com.centrewardz',
'com.boistarrewardz'
]

for app_name, app_id in zip(app_names, app_ids):

    start = dt.datetime.now(tz=get_localzone())
    fmt = "%m/%d/%y - %I:%M:%S %p"
    print('-------------------------------------------------------------------------')
    print(f'{app_name} started at {start.strftime(fmt)}\n')

    try:
        app_reviews = []
        count = 200
        batch_num = 0

        rvws, token = reviews(app_id, lang='en', country='us', sort=Sort.NEWEST, count=count)
        if not rvws:
            print(f"No reviews found for {app_name} — app may not exist or have no reviews.\n")
            continue

        for r in rvws:
            r['app_name'] = app_name
            r['app_id'] = app_id
        app_reviews.extend(rvws)
        batch_num += 1
        print(f'Batch {batch_num} completed.')
        time.sleep(random.randint(1, 5))

        pre_review_ids = [r['reviewId'] for r in app_reviews]

        for batch in range(4999):
            rvws, token = reviews(app_id, lang='en', country='us', sort=Sort.NEWEST, count=count, continuation_token=token)
            new_review_ids = []

            for r in rvws:
                r['app_name'] = app_name
                r['app_id'] = app_id
                new_review_ids.append(r['reviewId'])

            app_reviews.extend(rvws)
            batch_num += 1

            all_review_ids = pre_review_ids + new_review_ids
            if len(set(pre_review_ids)) == len(set(all_review_ids)):
                print(f'No reviews left to scrape. Completed {batch_num} batches.\n')
                break

            pre_review_ids = all_review_ids

            if batch_num % 100 == 0:
                print(f'Batch {batch_num} completed.')
                review_collection.insert_many(app_reviews)
                app_reviews = []

            time.sleep(random.randint(1, 5))

        print(f'Done scraping {app_name}.')
        print(f'Scraped a total of {len(set(pre_review_ids))} unique reviews.\n')

        if app_reviews:
            review_collection.insert_many(app_reviews)

        end = dt.datetime.now(tz=get_localzone())
        print(f"\nAll {app_name} reviews inserted at {end.strftime(fmt)}.")
        print(f"Time taken: {end - start}")
        print('-------------------------------------------------------------------------\n')
        time.sleep(random.randint(1, 5))

    except Exception as e:
        print(f"Failed to scrape {app_name} ({app_id}): {str(e)}\n")
        continue

# converting the results into dataframe
app_reviews_df = pd.DataFrame(list(review_collection.find({})))

# save to CSV
app_reviews_df.to_csv('playstore_bank_reviews.csv', index=False)
