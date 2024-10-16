import combined_analysis

def main():
    # Ask user for dataset path
    file_path = input("Enter the path to your dataset (CSV format): ")
    
    # Load dataset
    data = combined_analysis.load_data(file_path)
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Plot variable distribution")
        print("2. Conduct ANOVA")
        print("3. Conduct t-Test")
        print("4. Conduct chi-Square")
        print("5. Conduct Regression")
        print("6. Conduct Sentiment Analysis")
        print("7. Quit")
        
        choice = input("Enter your choice (1-7): ")
        
        if choice == '1':
            combined_analysis.plot_variable_distribution(data)
        elif choice == '2':
            combined_analysis.perform_anova(data)
        elif choice == '3':
            combined_analysis.perform_ttest(data)
        elif choice == '4':
            combined_analysis.perform_chi_square(data)
        elif choice == '5':
            combined_analysis.perform_regression(data)
        elif choice == '6':
            combined_analysis.analyze_sentiment(data)
        elif choice == '7':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
