import streamlit as st
from src.pages.UsedCar import UsedCarPrice as Price
from src.pages.Home import Home as Home


PAGES={'About me':Home,'Price Prediction':Price}
def write_page(page):  # pylint: disable=redefined-outer-name
    """Writes the specified page/module
    Our multipage app is structured into sub-files with a `def write()` function
    Arguments:
        page {module} -- A module with a 'def write():' function
    """
    # _reload_module(page)
    page.write()
def main():
    st.sidebar.title("Projects")
    choice=st.sidebar.radio("Explore the Projects below ?",tuple(PAGES.keys()))
    if choice ==None:
        Home.write()
    else:
        page=PAGES[choice]
        with st.spinner(f"Loading {choice} ..."):
            write_page(page)
    
if __name__ == "__main__":
    main()
    

