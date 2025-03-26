if __name__ == "__main__":
    # Run the GUI
    root.mainloop()

    # Optional: Schedule the crawler if not using external tools
    import schedule
    import time

    def run_crawler():
        print("Running the crawler...")
        parse_research_papers()
        save_to_json()

    schedule.every().sunday.at("00:00").do(run_crawler)

    while True:
        schedule.run_pending()
        time.sleep(1)