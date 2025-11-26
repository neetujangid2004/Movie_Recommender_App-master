import streamlit as st
from PIL import Image
import json
from Classifier import KNearestNeighbours
from bs4 import BeautifulSoup
import requests, io
import PIL.Image
from urllib.request import urlopen

# -------------------------------
# Load dataset
# -------------------------------
with open('./Data/movie_data.json', 'r+', encoding='utf-8') as f:
    data = json.load(f)
with open('./Data/movie_titles.json', 'r+', encoding='utf-8') as f:
    movie_titles = json.load(f)

# -------------------------------
# Request headers for IMDb
# -------------------------------
hdr = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/128.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9'
}

# -------------------------------
# Fetch movie poster safely
# -------------------------------
@st.cache_data(show_spinner=False)
def fetch_movie_poster(imdb_link):
    try:
        if not imdb_link.startswith("http"):
            imdb_link = "https://www.imdb.com" + imdb_link

        response = requests.get(imdb_link, headers=hdr, timeout=10)
        if response.status_code != 200:
            return "https://via.placeholder.com/158x301?text=No+Image"

        soup = BeautifulSoup(response.text, 'html.parser')
        imdb_dp = soup.find("meta", property="og:image")
        if imdb_dp and 'content' in imdb_dp.attrs:
            return imdb_dp['content']
        else:
            return "https://via.placeholder.com/158x301?text=No+Image"
    except Exception:
        return "https://via.placeholder.com/158x301?text=No+Image"

# -------------------------------
# Fetch movie info safely
# -------------------------------
@st.cache_data(show_spinner=False)
def get_movie_info(imdb_link):
    try:
        if not imdb_link.startswith("http"):
            imdb_link = "https://www.imdb.com" + imdb_link

        response = requests.get(imdb_link, headers=hdr, timeout=10)
        if response.status_code != 200:
            return (
                "Director: Not available",
                "Cast: Not available",
                "Story: Not available",
                "Total Rating count: N/A"
            )

        soup = BeautifulSoup(response.text, 'html.parser')
        imdb_content = soup.find("meta", property="og:description")
        movie_descr = imdb_content['content'].split('.') if imdb_content and 'content' in imdb_content.attrs else []

        movie_director = "Director: Not available"
        movie_cast = "Cast: Not available"
        movie_story = "Story: Not available"

        if len(movie_descr) > 0:
            movie_director = "Director: " + movie_descr[0].strip()
        if len(movie_descr) > 1:
            movie_cast = "Cast: " + movie_descr[1].replace('With', '').strip()
        if len(movie_descr) > 2:
            movie_story = "Story: " + movie_descr[2].strip() + '.'

        rating_tag = soup.find("span", class_="sc-bde20123-1 iZlgcd")
        movie_rating = 'Total Rating count: ' + rating_tag.text.strip() if rating_tag else 'Total Rating count: N/A'

        return movie_director, movie_cast, movie_story, movie_rating

    except Exception:
        return (
            "Director: Not available",
            "Cast: Not available",
            "Story: Not available",
            "Total Rating count: N/A"
        )

# -------------------------------
# KNN recommender
# -------------------------------
def KNN_Movie_Recommender(test_point, k):
    target = [0 for _ in movie_titles]
    model = KNearestNeighbours(data, target, test_point, k=k)
    model.fit()
    table = []
    for i in model.indices:
        table.append([movie_titles[i][0], movie_titles[i][2], data[i][-1]])
    return table

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Movie Recommender System")

def run():
    img1 = Image.open('./meta/logo.jpg')
    img1 = img1.resize((250, 250))
    st.image(img1, use_container_width=False)
    st.title("🎬 Movie Recommender System")
    st.markdown(
        '''<h4 style='text-align: left; color: #d73b5c;'>* Data is based on "IMDB 5000 Movie Dataset"</h4>''',
        unsafe_allow_html=True)

    genres = [
        'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama',
        'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror', 'Music', 'Musical',
        'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western'
    ]

    movies = [title[0] for title in movie_titles]
    category = ['--Select--', 'Movie based', 'Genre based']
    cat_op = st.selectbox('Select Recommendation Type', category)

    if cat_op == category[0]:
        st.warning('Please select Recommendation Type!!')

    elif cat_op == category[1]:
        select_movie = st.selectbox('Select movie (Recommendation will be based on this selection)', ['--Select--'] + movies)
        dec = st.radio("Want to Fetch Movie Poster?", ('Yes', 'No'))
        st.markdown(
            '''<h4 style='text-align: left; color: #d73b5c;'>* Fetching Movie Posters may take time.</h4>''',
            unsafe_allow_html=True)

        if select_movie == '--Select--':
            st.warning('Please select a Movie!!')
        else:
            no_of_reco = st.slider('Number of movies you want Recommended:', min_value=5, max_value=20, step=1)
            genres_vector = data[movies.index(select_movie)]
            test_points = genres_vector
            table = KNN_Movie_Recommender(test_points, no_of_reco + 1)
            table.pop(0)
            st.success('Some recommended movies:')

            for idx, (movie, link, ratings) in enumerate(table, start=1):
                st.markdown(f"### ({idx}) [{movie}]({link})")
                col1, col2 = st.columns([1, 3])

                with col1:
                    if dec == 'Yes':
                        poster_url = fetch_movie_poster(link)
                        st.image(poster_url, width=150)
                    else:
                        st.image("https://via.placeholder.com/150x225?text=No+Image", width=150)

                with col2:
                    director, cast, story, total_rat = get_movie_info(link)
                    st.markdown(f"**{director}**")
                    st.markdown(f"**{cast}**")
                    st.markdown(story)
                    st.markdown(total_rat)
                    st.markdown(f"**IMDB Rating:** {ratings} ⭐")

                st.markdown("---")

    elif cat_op == category[2]:
        sel_gen = st.multiselect('Select Genres:', genres)
        dec = st.radio("Want to Fetch Movie Poster?", ('Yes', 'No'))
        st.markdown(
            '''<h4 style='text-align: left; color: #d73b5c;'>* Fetching Movie Posters may take time.</h4>''',
            unsafe_allow_html=True)

        if sel_gen:
            imdb_score = st.slider('Choose IMDb score:', 1, 10, 8)
            no_of_reco = st.number_input('Number of movies:', min_value=5, max_value=20, step=1)
            test_point = [1 if genre in sel_gen else 0 for genre in genres]
            test_point.append(imdb_score)
            table = KNN_Movie_Recommender(test_point, no_of_reco)
            st.success('Some recommended movies:')

            for idx, (movie, link, ratings) in enumerate(table, start=1):
                st.markdown(f"### ({idx}) [{movie}]({link})")
                col1, col2 = st.columns([1, 3])

                with col1:
                    if dec == 'Yes':
                        poster_url = fetch_movie_poster(link)
                        st.image(poster_url, width=150)
                    else:
                        st.image("https://via.placeholder.com/150x225?text=No+Image", width=150)

                with col2:
                    director, cast, story, total_rat = get_movie_info(link)
                    st.markdown(f"**{director}**")
                    st.markdown(f"**{cast}**")
                    st.markdown(story)
                    st.markdown(total_rat)
                    st.markdown(f"**IMDB Rating:** {ratings} ⭐")

                st.markdown("---")

run()
