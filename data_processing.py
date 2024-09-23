import pandas as pd
import numpy as np
import timeit
import time 
import datetime 
import json
import re
import pickle
import spacy
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



def concatenate_matching_columns(df, pattern, col_name='concatenated'):
    regex = re.compile(pattern)
    matching_columns = [col for col in df.columns if regex.match(col)]
    
    if matching_columns:
        # Concatenate columns into a single column as strings
        df['concatenated'] = df[matching_columns].agg(lambda x: ', '.join(x.dropna().astype(str)),axis=1)
        
        # Remove whitespaces
#         df['concatenated'] = df['concatenated'].str.replace(r'\s+', '', regex=True)
        df.drop(matching_columns, axis=1, inplace=True)
        df.rename(columns={'concatenated':col_name}, inplace=True)
    
    return df


def remove_extra_columns(df, pattern):
    regex = re.compile(pattern)
    matching_columns = [col for col in df.columns if regex.match(col)]
    
    if matching_columns:
        matching_columns = [col for col in matching_columns if col not in \
                            [*concat_col_names,
                             "visible_instructors_0_title",
                             "visible_instructors_0_job_title",
                             "bestseller_badge_content_context_info_category_title",
                             "bestseller_badge_content_context_info_label_title",
                             "primary_category_title",
                             "primary_subcategory_title",
                             "context_info_label_title",
                             "price_detail_amount",
                             "quality_review_process_score",
                            ]
                           ]
        df.drop(matching_columns, axis=1, inplace=True)
        
    return df


def find_columns_with_string(df, pattern):
    regex = re.compile(pattern)
    matching_columns = [col for col in df.columns if regex.match(col)]
#     search_string_lower = search_string.lower()
#     matching_columns = [col for col in df.columns if search_string_lower in col.lower()]
    return matching_columns


def find_single_value_columns(df):
    single_value_columns = [col for col in df.columns if df[col].nunique() == 1]
    return single_value_columns


def missing_percentage(df):
    # Calculate the percentage of missing values for each column
    missing_percent = df.isnull().mean() * 100
    
    # Create a DataFrame to store the results
    missing_percent_df = pd.DataFrame(missing_percent, columns=['missing_percentage'])
    
    # Sort the DataFrame by the percentage of missing values in descending order
    missing_percent_df = missing_percent_df.sort_values(by='missing_percentage', ascending=False)
    
    return missing_percent_df


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove newlines and extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Process with spaCy
    doc = nlp(text)
    
    # Remove stopwords, punctuation, and perform lemmatization
    tokens = [token.lemma_.strip() for token in doc if not token.is_stop and not token.is_punct and token.is_alpha and token.lemma_ != "-PRON-"]
    
    # Join tokens back into a string
    return " ".join(tokens)


if __name__ == "__main__":

    udemy_courses_raw = pd.read_csv("udemy_courses.csv")

    udemy_courses = (
        udemy_courses_raw
        .loc[udemy_courses_raw["locale_simple_english_title"] == "English"]
        .rename(columns={"rating_distribution_0_count":"rating_1_star_count",
                        "rating_distribution_1_count":"rating_2_star_count",
                        "rating_distribution_2_count":"rating_3_star_count",
                        "rating_distribution_3_count":"rating_4_star_count",
                        "rating_distribution_4_count":"rating_5_star_count"
                        }
            )
        .copy()
    )

    patterns = [r"caption_locales_\d+_english_title",
            r"objectives_\d+",
            r"prerequisites_\d+",
            r"target_audiences_\d+",
            r"course_has_labels_\d+_label_title",
            r"requirements_data_items_\d+",
    ]
    
    concat_col_names = ["caption_languages",
                    "objectives",
                    "prerequisites",
                    "target_audiences",
                    "course_labels"
    ]
    
    for pattern, col_name in zip(patterns, concat_col_names):
        udemy_courses_1 = concatenate_matching_columns(udemy_courses, pattern, col_name) 

    # Specify the string to search for in column names
    search_string = ".+class"

    # Find and print columns that contain the search string
    matching_columns = find_columns_with_string(udemy_courses_1, search_string)

    # Find columns that contain only one unique value
    single_value_columns = find_single_value_columns(udemy_courses_1)

    single_value_columns.remove("locale_simple_english_title")

    udemy_courses_1.drop(single_value_columns, axis=1, inplace=True)

    missing_values = missing_percentage(udemy_courses_1)

    # Drop columns that have 100% NAs
    all_rows_missing_cols = list(missing_values[missing_values['missing_percentage'] == 100].index)

    udemy_courses_1.drop(all_rows_missing_cols, axis=1, inplace=True)

    drop_cols = ["url",
             "published_title",
             "locale_locale",
             "locale_title",
             "locale_english_title",
             "rating",
             "num_quizzes",
             "num_lectures",
             "num_curriculum_items",
             "num_published_practice_tests",
             "status_label",
             "created",
             "instructional_level",
             "content_info",
             "is_available_on_google_app",
             "is_available_on_ios",
             "apple_in_app_purchase_price_text",
             "is_cpe_compliant",
             "is_marketing_boost_agreed",
             "content_length_video",
             "has_sufficient_preview_length",
             "has_labs_in_course_prompt_setting",
             "is_organization_eligible",
             "is_language_course",
             "is_coding_exercises_badge_eligible",
             "is_course_in_ub_ever",
    ]   

    udemy_courses_1.drop(drop_cols, axis=1, inplace=True)

    udemy_courses_1.reset_index(drop=True, inplace=True)

    udemy_courses_1 = udemy_courses_1[(udemy_courses_1["estimated_content_length"] > 0) & (udemy_courses_1["last_update_date"].notnull())].copy()

    udemy_courses_1.drop_duplicates(inplace=True)
    
    # Create Categories
    # Use np.select to categorize the course duration
    conditions = [
        (udemy_courses_1['estimated_content_length'] <= 120),
        (udemy_courses_1['estimated_content_length'] > 120) & (udemy_courses_1['estimated_content_length'] <= 300),
        (udemy_courses_1['estimated_content_length'] > 300) & (udemy_courses_1['estimated_content_length'] <= 600),
        (udemy_courses_1['estimated_content_length'] > 600) & (udemy_courses_1['estimated_content_length'] <= 900),
        (udemy_courses_1['estimated_content_length'] > 900)
    ]

    choices = ["0-2 Hours", "2-5 Hours", "5-10 Hours", "10-15 Hours", "15+ Hours"]

    # Apply the conditions and labels
    udemy_courses_1['duration'] = np.select(conditions, choices)      

    # Categories to Ordinal to Scaled
    # Course Categories

    conditions = [
        (udemy_courses_1["avg_rating"] >= 4.5)  & (udemy_courses_1["num_reviews"] >= 1000),
        (udemy_courses_1["avg_rating"] >= 4.5)  & (udemy_courses_1["num_reviews"] < 1000) & (udemy_courses_1["num_reviews"] >= 30),
        (udemy_courses_1["avg_rating"] < 4.5)  & (udemy_courses_1["avg_rating"] >= 4.0) & (udemy_courses_1["num_reviews"] >= 30),
        (udemy_courses_1["avg_rating"] < 4.0) & (udemy_courses_1["num_reviews"] >= 30),
        (udemy_courses_1["num_reviews"] < 30)
    ]

    choices = ["Top-Rated & Popular", "Top-Rated & Growing", "Well-Rated & Established", "Moderately Rated & Established", "New or Low-Engagement"]

    udemy_courses_1["course_categories"] = np.select(conditions, choices, default="other")

    # Weighted average rating for instructors by number of reviews

    instructor_avg = (
        udemy_courses_1
        .groupby("visible_instructors_0_title", as_index=False)
        .agg( 
            total_weighted_rating = ("avg_rating", lambda rating: (rating * udemy_courses_1.loc[rating.index, "num_reviews"]).sum()),
            total_reviews = ("num_reviews", "sum")
        )
    )

    instructor_avg['weighted_avg_rating'] = instructor_avg["total_weighted_rating"] / instructor_avg["total_reviews"]

    udemy_courses_1 = udemy_courses_1.merge(instructor_avg[["visible_instructors_0_title", "weighted_avg_rating", "total_reviews"]], on="visible_instructors_0_title")


    # Instructor Categories

    conditions = [
        (udemy_courses_1["weighted_avg_rating"] >= 4.5)  & (udemy_courses_1["total_reviews"] >= 2000),
        (udemy_courses_1["weighted_avg_rating"] >= 4.5)  & (udemy_courses_1["total_reviews"] < 2000) & (udemy_courses_1["total_reviews"] >= 30),
        (udemy_courses_1["weighted_avg_rating"] < 4.5)  & (udemy_courses_1["weighted_avg_rating"] >= 4.1) & (udemy_courses_1["total_reviews"] >= 30),
        (udemy_courses_1["weighted_avg_rating"] < 4.1) & (udemy_courses_1["total_reviews"] >= 30),
        (udemy_courses_1["total_reviews"] < 30)
    ]

    choices = choices = ["Top-Rated & Popular", "Top-Rated & Growing", "Well-Rated & Established", "Moderately Rated & Established", "New or Low-Engagement"]

    udemy_courses_1["instructor_categories"] = np.select(conditions, choices, default="other")
        


    udemy_courses_1["last_update_date"] = pd.to_datetime(udemy_courses_1["last_update_date"])

    # Define the reference date (date data was retrieved)
    current_date = pd.to_datetime("2024-06-28")

    # Define the cutoff dates
    six_months_ago = current_date - pd.DateOffset(months=6)
    one_year_ago = current_date - pd.DateOffset(years=1)

    # Define the conditions
    conditions = [
        (udemy_courses_1["last_update_date"] >= six_months_ago),
        (udemy_courses_1["last_update_date"] < six_months_ago) & (udemy_courses_1["last_update_date"] >= one_year_ago),
        (udemy_courses_1["last_update_date"] < one_year_ago)
    ]

    # Define the corresponding labels
    choices = ["Within last 6 months", "6 months to 1 year", "More than 1 year"]

    # Apply the conditions and labels
    udemy_courses_1["last_updated"] = np.select(conditions, choices, default=None)

    ordinal_features = ["duration", "course_categories", "instructor_categories", "last_updated", "instructional_level_simple"]

    categories = [
        ["0-2 Hours", "2-5 Hours", "5-10 Hours", "10-15 Hours", "15+ Hours"],
        ["New or Low-Engagement", "Moderately Rated & Established", "Well-Rated & Established", "Top-Rated & Growing", "Top-Rated & Popular"],
        ["New or Low-Engagement", "Moderately Rated & Established", "Well-Rated & Established", "Top-Rated & Growing", "Top-Rated & Popular"],
        ["More than 1 year", "6 months to 1 year", "Within last 6 months"],
        ["Beginner", "All Levels", "Intermediate", "Expert"]
    ]

    ordinal_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(categories=categories)),
        ('scaler', MinMaxScaler())
    ])


    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('ord', ordinal_transformer, ordinal_features),
        ])

    # Fit and transform the data
    X_ordinal = preprocessor.fit_transform(udemy_courses_1) 

    with open("preprocessor.b", "wb") as f_out:
        pickle.dump(preprocessor, f_out)
    # Clean Features

    # Clean and convert the 'price' column
    # Courses are almost always discounted, just ask free vs paid
    udemy_courses_1['price'] = udemy_courses_1['price'].str.replace('$', '', regex=False)  # Remove the dollar sign
    udemy_courses_1['price'] = udemy_courses_1['price'].replace('Free', '0')  # Replace "Free" with "0"
    udemy_courses_1['price'] = pd.to_numeric(udemy_courses_1['price'])  # Convert to numeric type  


    # Force features to [0,1] scale
    boolean_features = ["is_paid", "has_certificate", "quizzes", "assignments", "bestseller"]
    udemy_courses_1["is_paid"] = udemy_courses_1["is_paid"].astype(int)
    udemy_courses_1["has_certificate"] = udemy_courses_1["has_certificate"].astype(int)
    udemy_courses_1["has_closed_caption"] = udemy_courses_1["has_closed_caption"].astype(int)
    udemy_courses_1["quizzes"] = np.where(udemy_courses_1["num_published_quizzes"] == 0, 0, 1)
    udemy_courses_1["assignments"] = np.where(udemy_courses_1["num_assignments"] == 0, 0, 1)
    udemy_courses_1["bestseller"] = np.where(udemy_courses_1["bestseller_badge_content_context_info_label_title"].isnull(), 0, 1)


    X_tabular = np.hstack((udemy_courses_1[boolean_features].values, X_ordinal))
    np.save("x_tabular", X_tabular)

    # Preprocess Text

    # Load spaCy model
    # Run this if you don't have it installed: python -m spacy download en_core_web_lg
    nlp = spacy.load("en_core_web_lg")


    text_columns = [
    'title', 'description', 'headline', 'context_info_label_title',
    'objectives', 'prerequisites', 'target_audiences', 'course_labels'
    ]

    udemy_courses_1[text_columns] = udemy_courses_1[text_columns].astype("str")

    # Apply preprocessing to specified columns and create new columns
    for col in text_columns:
        udemy_courses_1[f'{col}_clean'] = udemy_courses_1[col].apply(preprocess_text)

    # List of cleaned text columns to combine
    cleaned_text_columns = [
        'title_clean', 'description_clean', 'headline_clean', 'context_info_label_title_clean',
        'objectives_clean', 'prerequisites_clean', 'target_audiences_clean', 'course_labels_clean'
    ]

    # Combine the cleaned text fields efficiently
    udemy_courses_1['combined_text'] = udemy_courses_1[cleaned_text_columns].fillna('').agg(' '.join, axis=1)

    udemy_courses_1.to_csv('udemy_courses_features.csv', index=False)  

