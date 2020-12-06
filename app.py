import streamlit as st
import src.pages.Uber as Uber
import src.pages.UsedCarPrice as Price
import src.pages.Home as Home
import src.pages.first as First
import src.CarSpeed.terrain as Terrain

PAGES={'Crop Prediction':First,'About me':Home,'CarSpeedAutomation':Terrain,'Uber':Uber,'Price Prediction':Price}
def write_page(page):  # pylint: disable=redefined-outer-name
    """Writes the specified page/module
    Our multipage app is structured into sub-files with a `def write()` function
    Arguments:
        page {module} -- A module with a 'def write():' function
    """
    # _reload_module(page)
    page.write()
def main():
    st.set_page_config(page_title='Anush Projects', page_icon = "src/CarSpeed/favicon.ico")
    st.sidebar.title("Projects")
    choice=st.sidebar.radio("Explore the Projects below ?",tuple(PAGES.keys()))
    if choice ==None:
        First.write()
    else:
        page=PAGES[choice]
        with st.spinner(f"Loading {choice} ..."):
            write_page(page)
    
if __name__ == "__main__":
    main()
    

