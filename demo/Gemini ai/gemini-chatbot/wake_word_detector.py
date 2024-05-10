# Define your wake word
WAKE_WORD = "hey gemini"

def check_wake_word():
  """Prompts the user for input and checks for the wake word."""
  user_input = input("Enter your request: ")
  return user_input.lower() == WAKE_WORD

# Example usage
if __name__ == "__main__":
  if check_wake_word():
    print("Wake word detected! Launching main program...")
    # Import and run the main program (replace with your import statement)
    import main_program  # Assuming main_program.py exists
    main_program.run_gemini()  # Assuming run_gemini() is defined in main_program.py

