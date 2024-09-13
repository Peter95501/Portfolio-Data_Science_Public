# World Happiness Report Analysis

##### By: Peter Henry

##### URL: https://www.kaggle.com/datasets/unsdsn/world-happiness

# Intro:
"""
In this analysis, we examine the Kaggle dataset titled "World Happiness Report," which records happiness-related metrics for 155 countries. The happiness scores and rankings are derived from the Gallup World Poll, with the scores based on responses to a primary life evaluation question. This question, known as the Cantril Ladder, asks respondents to imagine a ladder where 10 represents the best possible life and 0 represents the worst possible life. Participants rate their current lives on this scale.

The dataset provides scores from nationally representative samples for the years 2013 to 2016, adjusted using Gallup's weighting system to ensure accurate representation. Following the happiness score, the dataset includes six factors—economic production, social support, life expectancy, freedom, absence of corruption, and generosity—that contribute to higher life evaluations in each country compared to Dystopia, a hypothetical country with the world’s lowest national averages for these factors. While these six factors do not affect the total happiness score, they offer insight into why some countries rank higher than others.

Features in this dataset include the following:
1. Country: The country being analyzed.
2. Region: The geographical region in which the country is located.
3. Happiness Rank: The ranking of the country based on its happiness score relative to other countries.
4. Happiness Score: The overall score representing the country's happiness level, derived from survey responses to the Cantril Ladder question, where individuals rate their lives on a scale from 0 (worst possible life) to 10 (best possible life).
5. Economy (GDP per Capita): A measure of the economic output per person in the country, reflecting its wealth and economic health.
6. Social Support: The extent to which individuals feel supported by their social network, including friends, family, and community.
7. Healthy life expectancy: The average number of years a person can expect to live in good health, based on current health conditions in the country.
8. Freedom to make life choices: A measure of individuals' perceived freedom to make decisions about their lives, such as personal and professional choices.
9. Perceptions of corruption: A measure of the perceived level of corruption in the government and businesses within the country.
10. Generosity: A measure of how charitable and giving the population is, based on donations and acts of kindness.
11. Dystopia Residual: A hypothetical measure used as a benchmark to compare each country's performance. It represents the gap between the worst possible life conditions (Dystopia) and the actual conditions in the country.
12. Year: The year the data was collected for that country.
"""

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def combine_world_happiness_data(directory):
    """
    Combines multiple World Happiness CSV files into a single dataframe.
    
    Params:
    directory (str): The directory path where the CSV files are located.
    
    Returns:
    pd.DataFrame: A combined dataframe with standardized columns and a year column.
    """

    # Preprocess

    # Define standard column names for unification
    standard_columns = {
        'Country': 'Country',
        'Region': 'Region',
        'Happiness Rank': 'Happiness Rank',
        'Happiness Score': 'Happiness Score',
        'Standard Error': 'Standard Error',
        'Economy (GDP per Capita)': 'Economy (GDP per Capita)',
        'Family': 'Social support',
        'Health (Life Expectancy)': 'Healthy life expectancy',
        'Freedom': 'Freedom to make life choices',
        'Trust (Government Corruption)': 'Perceptions of corruption',
        'Generosity': 'Generosity',
        'Dystopia Residual': 'Dystopia Residual',
        'Lower Confidence Interval': 'Lower Confidence Interval',
        'Upper Confidence Interval': 'Upper Confidence Interval',
        'Economy..GDP.per.Capita.': 'Economy (GDP per Capita)',
        'Health..Life.Expectancy.': 'Healthy life expectancy',
        'Trust..Government.Corruption.': 'Perceptions of corruption',
        'Overall rank': 'Happiness Rank',
        'Country or region': 'Country',
        'Score': 'Happiness Score',
        'GDP per capita': 'Economy (GDP per Capita)',
        'Social support': 'Social support',
    }

    # Automatically find all CSV files in the specified directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    # Extract the year from the file name and load each file
    dataframes = {}
    for file_name in csv_files:
        year = file_name.split('.')[0]  # Extract year from filename
        file_path = os.path.join(directory, file_name)
        df = pd.read_csv(file_path)
        df.rename(columns=standard_columns, inplace=True)
        df['Year'] = year  # Add year column
        # Add missing columns with NaN values
        for col in standard_columns.values():
            if col not in df.columns:
                df[col] = pd.NA
        dataframes[year] = df

    combined_df = pd.concat(dataframes.values(), ignore_index=True)

    return combined_df

directory = r"C:\Users\peter\OneDrive\Documents\Portfolio - Data Science\Extracurricular\Datasets\World Happiness"

combined_df = combine_world_happiness_data(directory)


# Lets look to see what has been created
print(combined_df.head(10))
print(combined_df.tail(10))

## Next Steps
#- Looks like there are some columns I do not want such as 'Standard Error', Lower Confidence Interval', 'Upper Confidence Interval', 'Whisker.high', 'Whisker.low'.
#- There are some columns that need merging like Happiness.Rank, Happiness.Score, Dytopia.Residual.
#- Finally, I want 'Region to be available for all years so we will need to come up with a solution on how to add those regions in dependant on what country it is.

columns_to_drop = [
    'Standard Error',
    'Lower Confidence Interval',
    'Upper Confidence Interval',
    'Whisker.high',
    'Whisker.low'
    ]

# Source column > Target column
column_pairs_to_merge = [
    ('Happiness.Rank', 'Happiness Rank'),
    ('Happiness.Score','Happiness Score'),
    ('Dystopia.Residual','Dystopia Residual')
    ]

def merge_multiple_columns(df, column_drop, column_pairs):
    """
    Merge values from multiple source columns into target columns and drop source columns.

    Params: 
    - df(pd.DataFrame): pd df
    - column_pairs(list of tuples): list of tuples containing two column names (source > target).
                                    Source column will be merged into target column 
    
    Returns:
    - pd.DataFrame: Updated dataframe with merged columns
    """
    df.drop(column_drop, axis=1, inplace=True)

    for source_col, target_col in column_pairs:
        df[target_col] = df[target_col].combine_first(df[source_col])

    df.drop(columns=[source_col for source_col, target_col in column_pairs], inplace=True)
    
    return df


combined_df = merge_multiple_columns(combined_df, columns_to_drop, column_pairs_to_merge)

print(combined_df.columns)

#Looks like we dropped and merged the data successfully! Now lets add those regions to the countries. Also I want to add a unique identifier to the country so we can identify a country based on an ID

# Checking to see if 'Year' is numeric or a string
print(combined_df['Year'].info())

def add_country_id(df):
    country_id_mapping = pd.factorize(df['Country'])[0] + 1 
    df['Country ID'] = country_id_mapping
    return df

def create_region_mapping(df_2015): # using year 2015 due to having the necessary information
    region_mapping = df_2015.set_index('Country')['Region'].to_dict()
    return region_mapping

def fill_missing_regions(df, region_mapping):
    df['Region'] = df['Region'].combine_first(df['Country'].map(region_mapping))
    return df

df_2015 = combined_df[combined_df['Year'] == '2015']
region_mapping = create_region_mapping(df_2015)

combined_df = fill_missing_regions(combined_df, region_mapping)
combined_df = add_country_id(combined_df)

print(combined_df[combined_df['Country'] == 'Switzerland'].head())

# Great! The unique ID for each country worked! Now lets check to see how regions went.
# - First thing to look at is if there are NA values

#Checking NA values for Region. These are countries that were not part of 2015
combined_df.isna().sum()

# Looking at the Countries and Country IDs that have NA values 
missing_regions = combined_df[combined_df['Region'].isna()][['Country','Country ID']].drop_duplicates()
print(missing_regions)

# Looking at the regions to add to each country in the above missing_regions
regions = combined_df['Region'].drop_duplicates().dropna()
print(f"Number of Regions: {regions.count()}\n\nRegions are:\n{regions}")

# Now that we have the countries with missing regions, and we have the regions we can apply our knowledge to apply the region to the country. Below is a dictionary related to that.

# Key is Country ID, value is applicable region
region_mapping_for_missing = {
    165: 'Eastern Asia',
    160: 'Latin America and Caribbean',
    166: 'Eastern Asia',
    161: 'Sub-Saharan Africa',
    163: 'Sub-Saharan Africa',
    164: 'Sub-Saharan Africa',
    167: 'Latin America and Caribbean',
    168: 'Middle East and Northern Africa',
    169: 'Central and Eastern Europe',
    170: 'Sub-Saharan Africa'
}

def add_missing_region(df, missing_regions):
    df['Region'] = df.apply(
        lambda row: region_mapping_for_missing[row['Country ID']]
            if pd.isna(row['Region']) and row['Country ID'] in region_mapping_for_missing
            else row['Region'],
        axis=1
    )
    return df

combined_df = add_missing_region(combined_df, region_mapping_for_missing)

print(f" NA values in Region: {combined_df['Region'].isna().sum()}")

# Yay we were able to apply regions based on the Country ID. Now lets look at the other NA values

print(combined_df.isna().sum())

#Dystopia Residual is almost a nonsensical value and it a feauture engineering example already entered into the dataset. In this dataset the dystopia value is supposed to represent the absolute minimum value for any feature. 

#The values that are missing can be calculated.
# - To calculate:
#     - round(Happiness score - sum(Economy, Social support, Healthy Life expect, Freedom, Perception of corruption, Generosity),3)

# Lets Calculate and add into your df. Also, lets take care of the one perception of corruption. We will look further into what we should do for the 1 Na value

def fill_missing_dystopia(df):
    df['Dystopia Residual'] = df.apply(
        lambda row: round(row['Happiness Score'] - sum([
            row['Economy (GDP per Capita)'],
            row['Social support'],
            row['Healthy life expectancy'],
            row['Freedom to make life choices'],
            row['Perceptions of corruption'] if not pd.isna(row['Perceptions of corruption']) else 0,
            row['Generosity']
        ]),3) if pd.isna(row['Dystopia Residual']) else row['Dystopia Residual'], axis=1
    )
    return df

combined_df = fill_missing_dystopia(combined_df)

print(combined_df.isna().sum())

# Looking at the original missing values based on the Country ID
missing_country_ids = [165, 160, 166, 161, 163, 164, 167, 168, 169, 170]
filtered_df = combined_df[combined_df['Country ID'].isin(missing_country_ids)]
print(filtered_df[['Country','Region']].drop_duplicates())

# Great! All the missing regions have been filled with what we wanted!!! Now to look at the pesky Prection of corruption NA value

print(f"Below is the row with the missing value for Perception of Corruption:\n\n{combined_df[combined_df['Perceptions of corruption'].isna()]}")

# Lets see if Perception of Corruption is available for the UAE in other years. If so lets take the average of all the available years and add that into the 2018 year. 

print(combined_df[combined_df['Country ID'] == 20][['Country ID','Year','Perceptions of corruption']].dropna())

# So we can see that 2018 is the only year missing. Lets take the average of all the other years. 2019 looks low but lets assume that is an odd year out since we do not have any more information. We will take the average of the years 2015, 2016, and 2017 without 2019.

# some simple commands to get the one value out
uae_data = combined_df[combined_df['Country ID'] == 20][['Year','Perceptions of corruption']].dropna()
uae_avg = uae_data[uae_data['Year'].isin(['2015','2016','2017'])]['Perceptions of corruption'].mean()
combined_df.loc[(combined_df['Country ID'] == 20) & (combined_df['Year'] == '2018'), 'Perceptions of corruption'] = round(uae_avg,5)

print(combined_df[combined_df['Country ID'] == 20][['Country ID','Year','Perceptions of corruption']])

# The Average has been added in. 

## Now lets look at some descriptive statistics to see if there is anything off.

print(combined_df.describe())

# There are a couple areas that look off.
# Any minimum at 0.000000 look to be off. To me, no country should have a 0 value as that would be considered at or below the dystopia value.
# So we will probably need to correct the zeros for min and then recalc the Dystopia residual like we did for the 213 missing Dystopia Residual values. Since there is proprietary values for each Dystopia Residual in 
# each row, it is impossible to accurately calculate to the 6th digit place. But we can be accurate to the 3rd digit place. We will round all Dystopia Residual values to the 1000ths decimal place in order to be accurate

# Lets start with Economy (GDP per Capita)

# Example of missing values in Economy GDP Per Capita
print(combined_df[combined_df['Economy (GDP per Capita)'] == 0])


# Pulling in the data for any country with 0 in Economy column
combined_df['Economy (GDP per Capita)'] = combined_df['Economy (GDP per Capita)'].astype(float)
zero_gdp_countries = combined_df[combined_df['Economy (GDP per Capita)'] == 0]['Country ID'].unique()
missing_GDP_country_data = combined_df[combined_df['Country ID'].isin(zero_gdp_countries)]
missing_GDP_country_data = missing_GDP_country_data.sort_values(by=['Country ID', 'Year'])
print(missing_GDP_country_data)

def avg_min_values(df):

    columns_to_correct = [
        'Economy (GDP per Capita)', 
        'Social support', 
        'Healthy life expectancy', 
        'Freedom to make life choices', 
        'Perceptions of corruption', 
        'Generosity'
    ]

    for column in columns_to_correct:
        zero_value_countries = df[df[column] == 0]['Country ID'].unique()

        for country_id in zero_value_countries:
            non_zero_values = df[(df['Country ID'] == country_id) & (df[column] != 0)][column]
            avg_value = non_zero_values.mean()
            df.loc[(df['Country ID'] == country_id) & (df[column] == 0), column] = avg_value

    return df

combined_df = avg_min_values(combined_df)  

print(combined_df.describe())

combined_df.to_csv('C:/Users/peter/OneDrive/Documents/Portfolio - Data Science/Extracurricular/Coding/World Happiness/cleaned_dataset.csv',index=False)

# EDA
## Descriptive Stats

print(combined_df.describe())

save_path = "C:/Users/peter/OneDrive/Documents/Portfolio - Data Science/Extracurricular/Coding/World Happiness/"

plt.figure(figsize=(10,6))
sns.histplot(combined_df['Happiness Score'], bins=30, kde=True)
plt.title('Distribution of Happiness Scores')
plt.xlabel('Happiness Score')
plt.ylabel('Frequency')
plt.savefig(f'{save_path}Distribution of Happiness Scores.png')
plt.show()


plt.figure(figsize=(12,8))
sns.boxplot(x='Region', y='Happiness Score', data=combined_df)
plt.xticks(rotation=90)
plt.title('Region-wise Happiness Score Distribution')
plt.xlabel('Region')
plt.ylabel('Happiness Score')
plt.tight_layout()
plt.savefig(f'{save_path}Region-wise Happiness Score Distribution.png', bbox_inches='tight')
plt.show()


top_10_countries = combined_df.groupby('Year').apply(lambda x: x.nlargest(10, 'Happiness Score')).reset_index(drop=True)['Country'].unique()
filtered_df = combined_df[combined_df['Country'].isin(top_10_countries)]

plt.figure(figsize=(14,8))
sns.barplot(x='Happiness Score', y='Country', hue='Year', data=filtered_df)
plt.title('Top 10 Happiest Countries by Year (Including all top countries)')
plt.xlabel('Happiness Score')
plt.ylabel('Country')
plt.savefig(f'{save_path}Top 10 Happies Countries by Year.png')
plt.show()


top_10 = combined_df.groupby('Year').apply(lambda x: x.nlargest(10, 'Happiness Score')).reset_index(drop=True)

plt.figure(figsize=(14,8))

for country in top_10['Country'].unique():
    country_data = top_10[top_10['Country'] == country]
    plt.plot(country_data['Year'], country_data['Happiness Score'], marker='o', label=country)

plt.title('Top 10 Happiest Countries by Year')
plt.xlabel('Year')
plt.ylabel('Happiness Score')
plt.xticks(rotation=45)
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f'{save_path}Top 10 Happies Countries by Year.png')
plt.show()


avg_happiness_per_year = combined_df.groupby('Year')['Happiness Score'].mean().reset_index()

plt.figure(figsize=(14,8))
plt.plot(avg_happiness_per_year['Year'], avg_happiness_per_year['Happiness Score'], marker='o')
plt.title('Average Happiness Across the Years')
plt.xlabel('Years')
plt.ylabel('Happiness Score')
plt.grid()
plt.tight_layout()
plt.savefig(f'{save_path}Average Happiness Across the Years.png')
plt.show()


plt.figure(figsize=(12,8))
correlation_matrix = combined_df[['Happiness Score', 'Economy (GDP per Capita)', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Perceptions of corruption', 'Generosity']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.tight_layout()  # Adjust the layout to make room for labels
plt.savefig(f'{save_path}Correlation Heatmap.png', bbox_inches='tight')  # Save with tight bounding box
plt.show()


sns.pairplot(combined_df[['Happiness Score', 'Economy (GDP per Capita)', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Dystopia Residual']])
plt.show()


plt.figure(figsize=(10,6))
sns.regplot(x='Economy (GDP per Capita)', y='Happiness Score', data=combined_df, scatter_kws={'s':20})
plt.title('Happiness Score vs Economy (GDP per Capita)')
plt.xlabel('Economy (GDP per Capita)')
plt.ylabel('Happiness Score')
plt.savefig(f'{save_path}Regression plot of Economy and Happiness Score.png')
plt.show()


plt.figure(figsize=(10,6))
sns.regplot(x='Economy (GDP per Capita)', y='Healthy life expectancy', data=combined_df, scatter_kws={'s':20})
plt.title('Happiness Score vs Healthy Life Expectancy)')
plt.xlabel('Economy (GDP per Capita)')
plt.ylabel('Happiness Score')
plt.savefig(f'{save_path}Regression plot of Economy and Healthly life expectancy.png')
plt.show()
