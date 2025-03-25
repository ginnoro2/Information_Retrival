import time
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk  # Added ttk for dropdown menu
import webbrowser
import json
import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import re  # For regex-based normalization

# Setup Selenium WebDriver
options = Options()
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option("useAutomationExtension", False)
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# URL of the research portal
URL = "https://pureportal.coventry.ac.uk/en/organisations/fbl-school-of-economics-finance-and-accounting/publications/"
driver.get(URL)

# Wait for the page to load
try:
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CLASS_NAME, "list-results")))
except:
    print("Timeout: Could not find research papers.")
    driver.quit()
    exit()

# Scroll down to load more research papers
scroll_pause_time = 2
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(scroll_pause_time)

    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break  # No more content to load
    last_height = new_height

# Extract full page content after loading
soup = BeautifulSoup(driver.page_source, "html.parser")
driver.quit()

# Storage for research papers
research_papers = []
inverted_index = {"title": {}, "author": {}}

# Function to normalize text (remove punctuation and convert to lowercase)
def normalize_text(text):
    # Remove punctuation and convert to lowercase
    return re.sub(r"[^\w\s]", "", text.lower())

# Function to extract research papers
def parse_research_papers():
    global research_papers, inverted_index
    results = soup.find_all("li", class_="list-result-item")

    for res in results:
        try:
            title_tag = res.find("h3", class_="title")
            if not title_tag:
                continue  # Skip if no title

            title = title_tag.text.strip()
            link_tag = title_tag.find("a")
            if not link_tag:
                continue  # Skip if no link

            link = link_tag["href"]
            if not link.startswith("http"):
                link = "https://pureportal.coventry.ac.uk" + link  # Ensure correct link format

            author_tag = res.find("a", class_="link person")
            author_name = author_tag.text.strip() if author_tag else "Unknown"

            date_tag = res.find("span", class_="date")
            publication_date = date_tag.text.strip() if date_tag else "Unknown"

            paper = {
                "title": title,
                "link": link,
                "author": author_name,
                "date": publication_date,
            }
            research_papers.append(paper)
            index_paper(title, author_name, paper)
        except Exception as e:
            print(f"Error parsing a research paper: {e}")
            continue

# Function to index papers by title and author
def index_paper(title, author, paper):
    # Index by title
    title_words = normalize_text(title).split()
    for word in title_words:
        if word not in inverted_index["title"]:
            inverted_index["title"][word] = []
        inverted_index["title"][word].append(paper)

    # Index by author
    normalized_author = normalize_text(author)
    author_words = normalized_author.split()
    for word in author_words:
        if word not in inverted_index["author"]:
            inverted_index["author"][word] = []
        inverted_index["author"][word].append(paper)

# Function to save research papers to a JSON file
def save_to_json(filename="research_papers.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(research_papers, f, indent=4, ensure_ascii=False)
    print(f"Data saved to {filename}")

# Function to save research papers to a CSV file
def save_to_csv(filename="research_papers.csv"):
    keys = ["title", "link", "author", "date"]  # Column headers
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(research_papers)
    print(f"Data saved to {filename}")

# Function to search research papers by title, author, or both
def search(query, search_type):
    normalized_query = normalize_text(query)  # Normalize the query
    query_words = normalized_query.split()
    results = set()

    if search_type == "Title" or search_type == "Both":
        # Search by title
        for word in query_words:
            if word in inverted_index["title"]:
                results.update(tuple(paper.items()) for paper in inverted_index["title"][word])

    if search_type == "Author" or search_type == "Both":
        # Search by author
        for word in query_words:
            if word in inverted_index["author"]:
                results.update(tuple(paper.items()) for paper in inverted_index["author"][word])

    return [dict(paper) for paper in results]

# Initialize Tkinter GUI
root = tk.Tk()
root.title("Research Paper Search")
root.geometry("800x600")

# Search Type Dropdown
search_type_label = tk.Label(root, text="Select Search Type:")
search_type_label.pack(pady=5)
search_type_var = tk.StringVar(value="Both")  # Default value
search_type_menu = ttk.Combobox(root, textvariable=search_type_var, values=["Title", "Author", "Both"], state="readonly")
search_type_menu.pack(pady=5)

# Search Query Entry
query_label = tk.Label(root, text="Enter search query:")
query_label.pack(pady=10)
query_entry = tk.Entry(root, width=50)
query_entry.pack()

# Result Display Area
result_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=20)
result_text.pack(pady=10)

# Function to display search results
def display_results():
    query = query_entry.get().strip()
    search_type = search_type_var.get()

    if not query:
        messagebox.showwarning("Input Error", "Please enter a search query.")
        return

    results = search(query, search_type)
    result_text.delete(1.0, tk.END)  # Clear previous results

    if not results:
        result_text.insert(tk.END, "No results found.\n")
        return

    for paper in results:
        result_text.insert(tk.END, f"Title: {paper['title']}\n")
        result_text.insert(tk.END, f"Author: {paper['author']}\n")
        result_text.insert(tk.END, f"Date: {paper['date']}\n")
        result_text.insert(tk.END, f"Link: {paper['link']}\n")
        result_text.insert(tk.END, "-" * 50 + "\n")

# Search Button
search_button = tk.Button(root, text="Search", command=display_results)
search_button.pack(pady=10)

# Function to save search results to a file
def save_results():
    query = query_entry.get().strip()
    search_type = search_type_var.get()

    if not query:
        messagebox.showwarning("Input Error", "Please enter a search query before saving.")
        return

    results = search(query, search_type)
    if not results:
        messagebox.showinfo("No Results", "No results found to save.")
        return

    # Ask user for file format
    file_format = messagebox.askquestion("Save Format", "Save as JSON (Yes) or CSV (No)?")
    if file_format == "yes":
        save_to_json("search_results.json")
    else:
        save_to_csv("search_results.csv")

# Save Results Button
save_button = tk.Button(root, text="Save Results", command=save_results)
save_button.pack(pady=10)

# Function to open a link when clicked
def open_link(event):
    try:
        cursor_index = result_text.index(tk.CURRENT)
        start = f"{cursor_index} linestart"
        end = f"{cursor_index} lineend"
        line = result_text.get(start, end)

        if "Link:" in line:
            link = line.split("Link:")[1].strip()
            webbrowser.open(link)
    except Exception as e:
        print(f"Error opening link: {e}")

# Bind click event to open links
result_text.bind("<Button-1>", open_link)

# Parse research papers and save initial data
parse_research_papers()
save_to_json()  # Save all extracted data to a JSON file by default

# Run the GUI
root.mainloop()