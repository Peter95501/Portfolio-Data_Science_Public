import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt





# SETUP OF STREAMLIT APP:

st.set_page_config(layout="wide")

image_path = "World_Happiness_Files/"

if 'page' not in st.session_state:
    st.session_state.page = 'Introduction'

# Retrieve theme colors
primaryColor = st.get_option("theme.primaryColor") or "#c2652f"
backgroundColor = st.get_option("theme.backgroundColor") or "#FFFFFF"
secondaryBackgroundColor = st.get_option("theme.secondaryBackgroundColor") or "#F0F2F6"
textColor = st.get_option("theme.textColor") or "#31333F"

def add_orange_line():
    st.markdown("<hr style='border:2px solid orange; margin-top: -10px;'>", unsafe_allow_html=True)

def add_dashed_grey_line():
    st.markdown("<hr style='border:2px dashed grey; margin-top: -10px;'>", unsafe_allow_html=True)

def add_small_dashed_grey_line():
    st.markdown("<hr style='border:1px dashed grey; margin-top: -10px; margin-bottom: -10x;'>", unsafe_allow_html=True)

# Add custom CSS for skills buttons
st.markdown(f"""
    <style>
    /* Skills buttons */
    .skillsButton {{
        border: 2px solid {primaryColor};
        background-color: transparent;
        color: {primaryColor};
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 8px;
        display: inline-block;
        transition: background-color 0.3s, color 0.3s, border 0.3s;
        margin: 5px;
    }}

    .skillsButton:hover {{
        background-color: {primaryColor} !important;
        color: {backgroundColor} !important;
        border: 2px solid {primaryColor} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# Create a horizontal navigation menu using streamlit-option-menu
selected = option_menu(
    menu_title=None,
    options=["Introduction","EDA", "Neural Network - MNIST","Contact"],
    icons=["house-door", "bar-chart-line", "diagram-3", "send"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": f"{backgroundColor}"},
        "icon": {"color": f"{primaryColor}", "font-size": "18px"},
        "nav-link": {
            "font-size": "18px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": f"{secondaryBackgroundColor}",
            "color": f"{textColor}",
        },
        "nav-link-selected": {
            "background-color": f"{primaryColor}",
            "color": f"{backgroundColor}",
        },
    }
)

# SETUP OF STREAMLIT APP COMPLETE 


# Functions for each page.
def introduction():
    # code for intro
    st.title("About Me")
    # Line below About me
    add_orange_line()
    col1, col2 = st.columns([2,1])
    with col1:
        # About me Content
        st.write("""
        After starting my career as a financial analyst, I found myself drawn to the world of data science, where my passion for 
        research and understanding how things work could truly thrive. The transition from finance to data science has been an 
        exciting journey, as I have discovered the power of data-driven decision-making and the vast potential of machine learning.
        
        What excites me the most about data science is the ability to dive deep into complex problems, unraveling insights that 
        help make sense of the world around us. Developing machine learning models brings me great joy as it allows me to 
        combine my analytical mindset with the technical skills I have developed along the way. 
        
        Whether it is building predictive models, analyzing trends, or experimenting with algorithms, I am constantly motivated by 
        the challenge and creativity that data science offers. My journey is driven by a love for learning and a desire to 
        make impactful contributions using data.
        """)
        # Personal Info:
        st.write("Name: Peter Henry")
        st.write("Location: Willing to move anywhere in the contiguous USA")
        st.write("Interests: Machine Learning, Data Science, Business Consulting")
        st.write("[LinkedIn](https://www.linkedin.com/in/peter-henry-783a88121/)")    
    
    with col2:
        # Profile Pic
        try:
            st.image('Streamlit_for_portfolio/Profile_pic_of_me.jpg', width=450)
        except FileNotFoundError:
            st.error("Profile picture not found. Please ensure the image is in the correct directory.")


    # Introduction Skills Section
    st.subheader("My Skills")
    add_orange_line()
    # Columns for skills
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="skillsButton">Python</div>', unsafe_allow_html=True)
        st.markdown('<div class="skillsButton">R</div>', unsafe_allow_html=True)
        st.markdown('<div class="skillsButton">SQL</div>', unsafe_allow_html=True)
        st.markdown('<div class="skillsButton">Keras</div>', unsafe_allow_html=True)
        st.markdown('<div class="skillsButton">Sklearn</div>', unsafe_allow_html=True)
        st.markdown('<div class="skillsButton">PyTorch</div>', unsafe_allow_html=True)
        st.markdown('<div class="skillsButton">TensorFlow</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="skillsButton">Matplotlib</div>', unsafe_allow_html=True)
        st.markdown('<div class="skillsButton">Seaborn</div>', unsafe_allow_html=True)
        st.markdown('<div class="skillsButton">Data-driven decision-making</div>', unsafe_allow_html=True)
        st.markdown('<div class="skillsButton">NLP</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="skillsButton">Data Cleaning</div>', unsafe_allow_html=True)
        st.markdown('<div class="skillsButton">Preprocessing</div>', unsafe_allow_html=True)
        st.markdown('<div class="skillsButton">Statistical Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="skillsButton">Time-Series Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="skillsButton">Git/Github</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="skillsButton">Data Wrangling</div>', unsafe_allow_html=True)
        st.markdown('<div class="skillsButton">Predictive Modeling</div>', unsafe_allow_html=True)
        st.markdown('<div class="skillsButton">Data Mining</div>', unsafe_allow_html=True)
        st.markdown('<div class="skillsButton">ETL Processes</div>', unsafe_allow_html=True)

def eda():
    st.html("<h1 style='text-align: center;'>Exploratory Data Analysis Example</h1>")
    
    add_orange_line()
    # Load the dataset
    try:
        df = pd.read_csv("World_Happiness_Files/cleaned_dataset.csv")
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'cleaned_dataset.csv' is in the correct directory.")
    else:
        # Show Dataset Overview
        st.html("<h2 style='text-align: center;'>World Happiness Report Interactive EDA</h2>")


    try:
        # Create three columns with specified relative widths
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.write("")  # Empty column for spacing of the sunflower pic
        
        with col2:
            st.image("World_Happiness_Files/Happy_Sunflower.jpg", use_column_width=True)
        
        with col3:
            st.write("")  # Empty column for spacing of the sunflower pic
    except FileNotFoundError:
        st.error("Image not found. Please ensure 'Happy_Sunflower.jpg' is in the correct directory.")
    else:
        st.html("<style>img {margin-bottom: 60px;}</style>")

        
        st.html("<h3 style='text-align: center;'>Dataset Overview</h3>")

        with st.expander("Introduction:"):
            st.write(
                """ 
                **World Happiness Report Analysis**

                By: Peter Henry

                URL: [Kaggle Dataset](https://www.kaggle.com/datasets/unsdsn/world-happiness)

                **Introduction:**
                
                In this analysis, we examine the Kaggle dataset titled "World Happiness Report," which records happiness-related metrics for 155 countries. The happiness scores and rankings are derived from the Gallup World Poll, with the scores based on responses to a primary life evaluation question. This question, known as the Cantril Ladder, asks respondents to imagine a ladder where 10 represents the best possible life and 0 represents the worst possible life. Participants rate their current lives on this scale.

                The dataset provides scores from nationally representative samples for the years 2013 to 2016, adjusted using Gallup's weighting system to ensure accurate representation. Following the happiness score, the dataset includes six factors—economic production, social support, life expectancy, freedom, absence of corruption, and generosity—that contribute to higher life evaluations in each country compared to Dystopia, a hypothetical country with the world’s lowest national averages for these factors. While these six factors do not affect the total happiness score, they offer insight into why some countries rank higher than others.
                """
            )

        with st.expander("Features in this dataset include the following:"):
            st.write(
                """
                1. **Country**: The country being analyzed.
                2. **Region**: The geographical region in which the country is located.
                3. **Happiness Rank**: The ranking of the country based on its happiness score relative to other countries.
                4. **Happiness Score**: The overall score representing the country's happiness level, derived from survey responses to the Cantril Ladder question.
                5. **Economy (GDP per Capita)**: A measure of the economic output per person in the country.
                6. **Social Support**: The extent to which individuals feel supported by their social network.
                7. **Healthy Life Expectancy**: The average number of years a person can expect to live in good health.
                8. **Freedom to Make Life Choices**: A measure of individuals' perceived freedom to make decisions about their lives.
                9. **Perceptions of Corruption**: A measure of the perceived level of corruption in the government and businesses.
                10. **Generosity**: A measure of how charitable and giving the population is.
                11. **Dystopia Residual**: A hypothetical measure used as a benchmark to compare each country's performance.
                12. **Year**: The year the data was collected for that country.
                """
            )

         
        st.html("<h3 style='text-align: center;'>World Happiness Report - Interactive Scatter Plot</h3>")

        x_axis = st.selectbox("Select X-axis for Scatter Plot", df.columns, index=df.columns.get_loc("Economy (GDP per Capita)"))
        y_axis = st.selectbox("Select Y-axis for Scatter Plot", df.columns, index=df.columns.get_loc("Happiness Score"))
        fig = px.scatter(df, x=x_axis, y=y_axis, color='Region', title=f'{y_axis} vs {x_axis}')
        st.plotly_chart(fig)

        # Show Interactive Dist of Happiness Hist Precomputed Visualizations
        st.html("<h3 style='text-align: center;'>Interactive Distribution of Happiness Histogram</h3>")
        bins = st.slider("Select Number of Bins", min_value=5, max_value=50, value=20)
        fig_hist = px.histogram(df, x='Happiness Score', nbins=bins)
        st.plotly_chart(fig_hist)

        # Additional Visualizations
        add_orange_line()
        st.html("<h2 style='text-align: center;'>Below are additonal EDA reporting examples</h2>")
        
        st.html("<h3 style='text-align: center;'>Region-wise Happiness Score Distribution</h3>")
        try:
            region_dist_img = Image.open(image_path + "Region-wise Happiness Score Distribution.png")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.write("") # spacing for pic to be centered
            with col2:
                st.image(region_dist_img)
            with col3:
                st.write("") # spacing for pic to be centered
        except FileNotFoundError:
            st.error("Image not found. Please ensure the image is in the correct directory.")
        
        with st.expander("Summary of Region-wise Happiness Score Distribution"):
            st.write(
                """
                The boxplot for ‘Region-wise Happiness Score Distribution’ provides insight into how happiness scores are distributed across different geographical regions. Here's a brief summary of what the plot shows:

                1.	Western Europe and North America have the highest median happiness scores, with relatively narrow ranges indicating that most countries in these regions score consistently high on happiness.
                2.	Australia and New Zealand also show a narrow distribution, with both regions achieving high happiness scores, similar to North America.
                3.	Regions like Latin America and the Caribbean, Southeastern Asia, and Central and Eastern Europe show wider distributions of happiness scores. This suggests a greater variability in the happiness levels among countries within these regions.
                4.	Sub-Saharan Africa and Southern Asia have the lowest median happiness scores, with Sub-Saharan Africa having a slightly wider range, indicating variability in happiness across countries within this region.
                5.	The plot also includes outliers in regions like Latin America and the Caribbean and Sub-Saharan Africa, which could indicate countries with unusually high or low happiness scores compared to others in their region.

                This plot highlights the significant differences in happiness levels across different global regions, with wealthier regions (like Western Europe and North America) generally scoring higher on happiness than less affluent regions (like Sub-Saharan Africa and Southern Asia)."
                """
            )

        st.html("<h3 style='text-align: center;'>Top 10 Happiest Countries by Year</h3>")
        try:
            top_countries_img = Image.open(image_path + "Top 10 Happies Countries by Year.png")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.write("") # spacing for pic to be centered
            with col2:
                st.image(top_countries_img)
            with col3:
                st.write("") # spacing for pic to be centered
        except FileNotFoundError:
            st.error("Image not found. Please ensure the image is in the correct directory.")
        
        with st.expander("Summar of Top 10 Happiest Countries by Year"):
            st.write(
                """
                The line plot titled Top 10 Happiest Countries by Year shows the trends in happiness scores for the top 10 happiest countries from 2015 to 2019. Here's a summary of the key insights from the plot:

                1.	Finland shows a strong upward trend in happiness, starting with a relatively low position in 2015 and eventually becoming the highest-ranked country by 2019.
                2.	Switzerland, Iceland, and Norway maintain consistently high happiness scores, though their trends show some fluctuations over the years. Switzerland and Iceland show slight declines, while Norway shows some variation but remains near the top.
                3.	Denmark and Canada have relatively stable happiness scores across the years, with minor fluctuations. Denmark remains consistently high in the rankings.
                4.	Netherlands shows a strong increase from 2017 onwards, reaching its highest score in 2019.
                5.	Sweden, New Zealand, Australia, and Austria show more variation. For example, Austria sees a noticeable drop in happiness scores from 2015 to 2019, while New Zealand remains mostly flat.
                6.	The range of happiness scores among these top 10 countries is fairly narrow, between 7.3 and 7.8, but the small shifts in scores can still lead to notable changes in rankings over time.

                This plot highlights the stability and upward trends in happiness for certain countries, such as Finland, while others experience minor fluctuations, reflecting the dynamic nature of happiness across the world’s top-ranked countries.
                """
            )


        st.html("<h3 style='text-align: center;'>Average Happiness Across the Years</h3>")
        try:
            avg_happiness_img = Image.open(image_path + "Average Happiness Across the Years.png")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.write("") # spacing for pic to be centered
            with col2:
                st.image(avg_happiness_img)
            with col3:
                st.write("") # spacing for pic to be centered
        except FileNotFoundError:
            st.error("Image not found. Please ensure the image is in the correct directory.")

        with st.expander("Average Happiness Across the Years"):
            st.write(
                """
                The line plot titled "Average Happiness Across the Years" shows the trend in the average happiness score for all countries from 2015 to 2019. Here’s a brief summary of what the plot reveals:

                1.	2015 to 2016: There is a slight increase in the average happiness score from 2015 to 2016, suggesting a positive trend in global happiness during this period.
                2.	2017: The average happiness score drops noticeably in 2017, reaching the lowest point in the observed period. This indicates a decline in the global average happiness level for that year.
                3.	2018 to 2019: After the decline in 2017, the happiness score rises sharply in both 2018 and 2019, with the highest average score recorded in 2019. This suggests a strong recovery in global happiness following the 2017 dip.
                
                Overall, the plot indicates some fluctuation in global happiness across these years, with 2017 standing out as a year of decline, followed by a sharp rebound through 2019.

                """
            )

        st.html("<h3 style='text-align: center;'>Correlation Matrix of Features</h3>")
        try:
            correlation_img = Image.open(image_path + "Correlation Heatmap.png")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.write("") # spacing for pic to be centered
            with col2:
                st.image(correlation_img)
            with col3:
                st.write("") # spacing for pic to be centered
        except FileNotFoundError:
            st.error("Image not found. Please ensure the image is in the correct directory.")

        with st.expander("Correlation Matrix of Features"):
            st.markdown(
            """
            The correlation matrix heatmap for the "Correlation Matrix of Features" visualizes the relationships between different factors that influence the happiness score. Here's a summary of the key insights:

            1. **Happiness Score**:
            - Strongly correlated with **Economy (GDP per Capita)** (0.79) and **Healthy Life Expectancy** (0.74). This indicates that countries with higher economic output and better life expectancy tend to have higher happiness scores.
            - Moderately correlated with **Social Support** (0.64) and **Freedom to Make Life Choices** (0.55), suggesting that social factors and personal freedom also play significant roles in increasing happiness.

            2. **Economy (GDP per Capita)**:
            - Strongly correlated with **Healthy Life Expectancy** (0.79) and moderately correlated with **Social Support** (0.57). This implies that wealthier countries tend to have better health outcomes and stronger social support systems.

            3. **Perceptions of Corruption**:
            - Only weakly correlated with **Happiness Score** (0.40). This suggests that while corruption perception plays a role in determining happiness, it is not as significant as factors like economy, health, and social support.

            4. **Generosity**:
            - Shows the weakest correlation with **Happiness Score** (0.14) and even negative correlations with some factors like **Economy (GDP per Capita)** and **Social Support**. This indicates that generosity, while positive, may not be a major determinant of happiness when compared to economic and health-related factors.

            Overall, the matrix highlights that **Economy (GDP per Capita)**, **Healthy Life Expectancy**, and **Social Support** are the most influential factors contributing to a higher happiness score, whereas **Generosity** and **Perceptions of Corruption** have relatively weaker impacts.
            """
        )

        add_orange_line()

        # Download World Happiness EDA Button
        col1, col2, col3 = st.columns([2,1,2])
        with col1:
            st.write("")
        with col2:
            with open("Streamlit_for_portfolio/archive/World_Happiness_EDA.html", "rb") as file:
                st.download_button(label="Download the EDA File", data=file, file_name="Peter_Henry_World_Happiness_EDA.html")
        with col3:
            st.write("")

def digit_rec():
    st.html("<h1 style='text-align: center;'>Neural Network for MNIST Digit Recognition </h1>")
    add_orange_line()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.write("")
    with col2:
        st.image("Digit_Recognizer/pics/header.png", use_column_width=True)
    with col3:
        st.write("")
    st.html("<h2 style='text-align: center;'>Project Description </h2>")
    st.write(
        """

        In this project, I developed a deep learning model to tackle the Kaggle Digit Recognizer competition, which focuses on the classic MNIST dataset. The goal was to create a neural network capable of accurately classifying handwritten digits (0 to 9) from grayscale images. This dataset, widely used in the machine learning community, serves as a benchmark for evaluating image classification models.

        Leveraging a fully connected neural network, I applied advanced techniques in data preprocessing and model tuning to optimize the performance of the classifier. Through iterative experimentation and optimization, my final model achieved a high level of accuracy, reaching a public leaderboard score of **0.99389**. This demonstrates the model's strong ability to generalize and recognize digit patterns with minimal error.

        This project not only showcases the power of neural networks in image recognition tasks but also reflects the critical steps in deep learning workflows, from model design to performance evaluation and submission.

        """
    )
    add_dashed_grey_line()

    ################# EDA
    nn_img_path = "Digit_Recognizer/pics/"

    st.html("<h3 style='text-align: center;'>NN EDA</h3>")

    col1, col2 = st.columns([1,1])
    with col1:
        st.html("<h3 style='text-align: center;'>Distribution of Digits in Training Data</h3>")
        st.write("""
                    The Distribution of Digits in the Training Data provides an essential view into how well-balanced the dataset is across each digit (0–9). Visualized as a bar plot, it shows the frequency of each digit within the dataset along with the percentage representation for added clarity.

                    In this visualization, we observe the following key insights:

                    - Digit '1' appears most frequently, making up 11.15% of the dataset.
                    - Digit '5' is the least represented, accounting for 9.04% of the total data.
                    - The distribution is quite balanced overall, with all digits falling between 9.04% and 11.15%, indicating a well-rounded dataset without major imbalances.

                    Why does this matter? A balanced dataset ensures that our model has sufficient examples of each digit during training, which helps it learn to accurately classify all digits, avoiding bias toward more frequent classes. This balance is crucial for the model's performance, as it ensures equitable learning across all digit categories.
                """)
    with col2:
        st.image(nn_img_path + "Distribution of Digits in Training Data.png", use_column_width=True)

    st.markdown("---")

    col1, col2 = st.columns([1,1])
    with col1:
        st.html("<h3 style='text-align: center;'>Distribution of Pixel Density</h3>")
        st.write("""
                    The histogram visualizes the Distribution of Pixel Intensities within the digit images. In grayscale images, pixel intensities range from 0 (black) to 255 (white), with intermediate values representing varying shades of gray.

                    Key observations from the plot:

                    - A significant portion of pixels has an intensity of 255, representing the white background in most images.
                    - The non-zero pixel intensities are distributed fairly evenly across the range from 1 to 250, which corresponds to the actual strokes of the digits.
                    - The low-frequency bars near zero represent the darkest pixels, which are used to draw the digit shapes themselves.

                    This visualization helps us understand the makeup of the images and ensures that the dataset contains a rich range of pixel intensities. A high concentration of white pixels (255) is expected due to the empty background in the digit images, while the spread of intensities reflects the presence of digit strokes in different shades.
                """)
    with col2:
        st.image(nn_img_path + "Distribution of Pixel Density.png", use_column_width=True)

    st.markdown("---")

    col1, col2 = st.columns([1,1])
    with col1:
        st.html("<h3 style='text-align: center;'>Sample Digits in Order</h3>")
        st.write("""
                    This visualization showcases one representative image for each digit (0-9) from the dataset, allowing us to visually inspect the variation in handwriting styles.

                    Key observations:

                    - The digits are clearly distinguishable, despite varying levels of thickness and curvature in the strokes, reflecting the diversity in how individuals write numbers.
                    - Some digits (like "1", "7", or "4") may have more variability in their forms, which could pose a challenge to the neural network during classification.

                """)
    with col2:
        st.image(nn_img_path + "Sample Digits in Order.png", use_column_width=True)

    st.markdown("---")

    col1, col2 = st.columns([1,1])
    with col1:
        st.html("<h3 style='text-align: center;'>Average Image for Each Digit (0-9)</h3>")
        st.write("""
                    The visualization above presents the average pixel intensity for each digit from 0 to 9 across the entire dataset. By averaging all instances of each digit, we generate a "blurred" or "fuzzy" version of the digit, which shows the most consistent patterns in the handwritten numbers.

                    Key observations:

                    - The overall structure of each digit is clear, especially for digits like "0," "1," and "8," which have distinct shapes.
                    - Some digits, like "2" and "5," show more variation, with a less defined appearance, suggesting more diversity in how individuals write these digits.
                    - This visualization is useful for understanding common patterns and how certain parts of the digits may vary or remain consistent across the dataset.
                """)
    with col2:
        st.image(nn_img_path + "Average Image for Each Digit (0-9).png", use_column_width=True)

    st.markdown("---")

    col1, col2 = st.columns([1,1])
    with col1:
        st.html("<h3 style='text-align: center;'>Pixel Variance Across All Images</h3>")
        st.write("""
                    The heatmap above visualizes the variance of pixel intensities across all the images in the dataset. In this context, variance refers to how much the pixel values differ across different images.

                    Key insights:

                    - The bright yellow center shows the highest variance, corresponding to the central area of the images where the digits are typically drawn. This indicates that this area has the most variation in pixel intensities across the different images.
                    - The darker areas toward the corners and edges indicate lower variance, as those regions are mostly empty or contain background pixels with consistent values close to 0 (black).
                    - This heatmap highlights that most of the variation occurs in the central region of the image, which aligns with where digits are typically located and drawn.

                    This variance analysis is useful in understanding which areas of the images contribute the most to the diversity in the dataset, and where a model needs to focus its learning for accurate digit classification.
                """)
    with col2:
        st.image(nn_img_path + "Pixel Variance Across All Images.png", use_column_width=True)

    st.markdown("---")

    ############ NN Code Snippet:
    st.html("<h3 style='text-align: center;'>NN Architecture Code Snippet </h3>")
    with st.expander("Code Snippet Dropdown:"):
        st.code(
            """
            # Libraries
            import torch
            import torch.nn as nn
            import torch.optim as optim

            # Neural Network Model Definition
            class Net(nn.Module):
                def __init__(self):
                    super(Net, self).__init__()
                    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
                    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                    self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
                    self.fc1 = nn.Linear(256, 512)
                    self.fc2 = nn.Linear(512, 256)
                    self.fc3 = nn.Linear(256, 10)
                    self.dropout = nn.Dropout(0.3)  # Dropout to prevent overfitting

                def forward(self, x):
                    x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
                    x = nn.functional.relu(nn.functional.max_pool2d(self.conv2(x), 2))
                    x = nn.functional.relu(nn.functional.max_pool2d(self.conv3(x), 2))
                    x = nn.functional.relu(nn.functional.max_pool2d(self.conv4(x), 2))
                    x = x.view(-1, 256)  # Flatten Tensor
                    x = nn.functional.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = nn.functional.relu(self.fc2(x))
                    x = self.fc3(x)
                    return x

            # Model Training
            net = Net().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience // 2, verbose=True)

            for epoch in range(50):
                net.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            """
        )

    ############### Model Arch:
    st.html("<h3 style='text-align: center;'>Model Architecture </h3>")
    st.write("""
        The neural network model used for the MNIST Digit Recognition task consists of four convolutional layers, followed by fully connected layers to classify the digit images. Here's a breakdown of the architecture:

        - **Input Layer**: The input consists of grayscale images of size 28x28 pixels.
        
        - **Convolutional Layers**:
            - Conv2d(1, 32): 32 filters with a kernel size of 3x3, stride of 1, and padding of 1, applied to the input image.
            - Conv2d(32, 64): 64 filters with the same kernel size and parameters.
            - Conv2d(64, 128): 128 filters.
            - Conv2d(128, 256): 256 filters.

        """)
    st.write("""    
        - **Pooling**: After each convolution, a 2x2 max pooling operation is applied, reducing the spatial dimensions of the image.

        - **Fully Connected Layers**:
            - Linear(256, 512): A fully connected layer with 512 units.
            - Linear(512, 256): A fully connected layer with 256 units.
            - Linear(256, 10): Output layer with 10 units, corresponding to the 10 digit classes (0-9).

        """)
    st.write("""
        - **Activation Function**: ReLU (Rectified Linear Unit) is applied after each convolutional and fully connected layer.

        - **Dropout**: A dropout layer with a 0.30 probability is used after the first fully connected layer to prevent overfitting.

        - **Output Layer**: The final layer uses a softmax activation (through cross-entropy loss) to classify the input into one of 10 digit categories.        
        
        """)
    add_dashed_grey_line()
    
    ################ Results & Perf:
    st.html("<h3 style='text-align: center;'>Results and Performance </h3>")
    st.write("""
        My neural network model for MNIST digit recognition achieved impressive results in the **Kaggle Digit Recognizer** competition. Below are the key performance metrics and results:
        """)
    
    st.write("""
        - **Kaggle Public Leaderboard Score**: **0.99389**
        - This score places the model among the top-performing entries for this task.
        - The model correctly predicts 99.39% of the digits in the test set.
        """)
    
    st.markdown("---")
    ########### RESULTS SECTION
    col1, col2 = st.columns([1,1])
    with col1:
        st.html("<h3 style='text-align: center;'>Training and Validation Loss over Epochs</h3>")
        st.write("""

                The plot above illustrates the Training Loss and Validation Loss across 14 epochs during the model's training process.
                
                - Training Loss (blue line) shows a steady and significant decrease as the model learns over time, dropping close to zero by the end of the training phase. This indicates that the model has successfully minimized errors on the training data, effectively learning from it.
                - Validation Loss (orange line) presents a more fluctuating pattern initially, indicating some variability when applied to unseen data. Around the 6th epoch, the validation loss stabilizes, showing consistent performance on the validation set.
                - The gap between training and validation loss widens slightly after the 6th epoch, indicating a minor risk of overfitting. However, this difference remains relatively small, suggesting that the model generalizes well to new data.
                - The use of early stopping and learning rate scheduling helped prevent overfitting and ensured that the training stopped at an optimal point.
                """)
    with col2:
        st.image(nn_img_path + "Training and Validation Loss over Epochs.png", use_column_width=True, )

    st.markdown("---")

    col1, col2 = st.columns([1,1])
    with col1:
        st.html("<h3 style='text-align: center;'>Training Accuracy Over Epochs</h3>")
        st.write("""
                This line chart shows how the Training Accuracy of the model improves over 14 epochs.
                
                - **Rapid Improvement**: The accuracy rises sharply from around 96% at the very first epoch to over 98% by the third epoch. This steep increase shows that the model is quickly learning to recognize the patterns in the training data.
                - **Near-Perfect Accuracy**: By the 8th epoch, the training accuracy approaches 100%, and it remains close to 100% for the remainder of the epochs. This indicates that the model has effectively learned to classify almost all digits correctly in the training set.
                
                - The near-perfect training accuracy showcases the strength of the model’s architecture, allowing it to fit well to the provided data. However, achieving high training accuracy alone does not necessarily guarantee generalization to unseen data, which is why comparing this with validation accuracy is crucial to ensure balanced performance.
                """)
    with col2:
        st.image(nn_img_path + "Training Accuracy Over Epochs.png", use_column_width=True, )

    st.markdown("---")

    col1, col2 = st.columns([1,1])
    with col1:
        st.html("<h3 style='text-align: center;'>Confusion Matrix & Accuracy Per Digit Class</h3>")
        st.write("""
                    The confusion matrix provides a detailed look into the performance of the model on each individual class (digit 0 to 9). Each row represents the actual digit, while each column represents the predicted digit. The diagonal entries show how often the model correctly predicted each digit, while the off-diagonal entries indicate misclassifications.
                    - The model performed very well across all digits, with minimal misclassifications. Most misclassifications are close to the diagonal, indicating the model occasionally predicts adjacent digits incorrectly (e.g., confusing a '9' for a '4').
                    - The matrix highlights that the model struggles slightly with certain digits like 2 and 9, which have a few more misclassifications compared to other digits.

                    In terms of accuracy by digit, the model performs very well:
                    - Most digits achieve over 99% accuracy, including 0, 1, 5, 6, and 8.
                    - The lowest accuracy is for digit 9 at 97.77%, likely due to its similarity with other digits like 4.
                    
                    These results below reflect a strong overall performance, with the model achieving an accuracy of 99.39%.
                """)
        st.image(nn_img_path + "Accuracy by Digit Class.png", width=250 )
    with col2:
        st.image(nn_img_path + "Confusion Matrix.png", use_column_width=True, )

    st.markdown("---")

    col1, col2 = st.columns([1,1])
    with col1:
        st.html("<h3 style='text-align: center;'>Misclassified Digits</h3>")
        st.write("""
                    These examples highlight the challenges in distinguishing between visually similar digits:

                    - 3 mistaken for 5: The curvatures of '3' and '5' are often confused, especially when the digit '5' is written with a rounder form.
                    - 4 confused with 9: Digits '4' and '9' share similarities in their structure, particularly when '4' is written with an open loop at the top, leading to potential misidentification.
                    - 7 misclassified as 2: The sharp angles of '7' can resemble the curves of '2', especially when the '7' is written without a distinct crossbar.
                """)
    with col2:
        st.image(nn_img_path + "Sample of Misclassified Digit Classes.png", use_column_width=True)
    
    add_dashed_grey_line()

    st.html("<h3 style='text-align: center;'>Submission to Kaggle </h3>")
    col1, col2 = st.columns([1,1])
    with col1:
        st.write("The submission achieved an impressive public score of 0.99389, placing it among the top performers in the leaderboard. This score reflects the model's ability to accurately predict handwritten digits from the test set provided by Kaggle.")
    with col2:
        st.image("Digit_Recognizer/pics/Digit_Recognizer_Code_Result_Accuracy.png")
        submission = pd.read_csv('Digit_Recognizer/output/submission_CNN_Digit_Recognizer_2024-09-24_15-00-07.csv') 

    st.markdown("""
                <div style='border: 2px solid orange; padding: 20px; margin-top: 50px; margin-bottom: 100px; text-align: center;'>
                    <h2 style='text-align: center;'>Conclusion</h2>
                    <h5 style='text-align: center;'>
                        This project has demonstrated the power of deep learning in recognizing handwritten digits with remarkable accuracy, achieving a score of 99.39% on Kaggle. However, there is always room for improvement, especially in addressing subtle misclassifications. By fine-tuning the model, increasing data augmentation, or exploring advanced architectures, we can push this accuracy even higher. Whether you are an aspiring data scientist or a seasoned professional, challenges like these present the perfect opportunity to sharpen your skills, dive deeper into machine learning, and contribute to the exciting world of AI-driven solutions. Now is the time to take action to explore, experiment, and innovate!
                    </h5>
                </div>
                """, unsafe_allow_html=True)
    
    st.html("<h2 style='text-align: center;'>Below are file downloads for the NN and submission files</h2>")
    st.markdown("<hr style='border:5px solid grey; margin-top: -10px;'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1,1])
    with col1:
        st.html("<h3 style='text-align: center;'>File Downloads </h3>")
        st.write("You can download an HTML version of the visualized jupyter notebook used to earn a 99.39% accuracy score. ")
        with open("Streamlit_for_portfolio/archive/Digit_Recognizer_NN_Code v2 testing Visualizations.html", "rb") as file:
            st.download_button(label="Download the NN File", data=file, file_name="Peter_Henry_Digit_Recognizer.html")
        st.write("Also, you can choose to download the submission file as a csv. ")
        with open("Digit_Recognizer/output/submission_CNN_Digit_Recognizer_2024-09-24_15-00-07.csv", "rb") as file:
            st.download_button(label="Download Submission file", data=file, file_name="Peter_Henry_submission_file.csv")
    with col2: 
        st.write("### Submission File Preview")
        st.dataframe(submission.head())
    


def contacts():
    st.title("Contacts")
    add_orange_line()
    st.write("**Email:** peter.henry.career@gmail.com")
    st.write("[LinkedIn Profile](https://www.linkedin.com/in/peter-henry-783a88121/)")
    st.write("[GitHub Profile](https://github.com/Peter95501)")



# Pages and function calls
if selected == "Introduction":
    introduction()
elif selected == "EDA":
    eda()
elif selected == "Neural Network - MNIST":
    digit_rec()
elif selected == "Contact":
    contacts()