# Rule-Based AI Chatbot
# Project 1 - DecodeLabs Industrial Training | Batch 2026
# Topic: Artificial Intelligence - Control Flow & Logic

# -------------------------------------------------------
# KNOWLEDGE BASE (Dictionary)
# Stores all the predefined intents and their responses.
# Using a dictionary gives us O(1) lookup time, which is
# much faster than a long if-elif ladder (O(n)).
# -------------------------------------------------------
responses = {
    # --- Greetings ---
    "hello"        : "Hey there! I'm DecoBot. How can I help you today?",
    "hi"           : "Hi! Nice to meet you. What can I do for you?",
    "hey"          : "Hey! What's up? Ask me anything.",
    "good morning" : "Good morning! Hope you're having a great day. How can I assist?",
    "good evening" : "Good evening! How can I help you tonight?",

    # --- How are you ---
    "how are you"  : "I'm just a bot, but I'm running perfectly! Thanks for asking.",
    "what's up"    : "Not much, just waiting to answer your questions!",
    "whats up"     : "Not much, just waiting to answer your questions!",

    # --- About the bot ---
    "who are you"  : "I'm DecoBot, a simple rule-based AI chatbot built for Project 1 at DecodeLabs.",
    "what are you" : "I'm a rule-based chatbot. I match your input to predefined rules and reply accordingly.",
    "what can you do" : "I can answer basic questions, greet you, tell you the time and date, do simple math, and more!",
    "your name"    : "My name is DecoBot! Nice to meet you.",

    # --- Help ---
    "help"         : "Sure! You can ask me things like: 'hello', 'how are you', 'what time is it', 'tell me a joke', or type 'exit' to quit.",

    # --- Time & Date ---
    "time"         : "__TIME__",      # special tag, handled in process_input()
    "what time is it" : "__TIME__",
    "date"         : "__DATE__",
    "what is today" : "__DATE__",
    "what day is it" : "__DATE__",
    "today"        : "__DATE__",

    # --- Jokes ---
    "tell me a joke" : "Why do programmers prefer dark mode? Because light attracts bugs! ",
    "joke"           : "I told my computer I needed a break... now it won't stop sending me Kit-Kat ads. ",
    "funny"          : "Why was the math book sad? Because it had too many problems!",

    # --- Motivational ---
    "motivate me"  : "Keep going! Every expert was once a beginner. You've got this! ",
    "motivation"   : "Success is not final, failure is not fatal — it's the courage to continue that counts!",
    "inspire me"   : "Dream big, code bigger. One line at a time. ",

    # --- Weather (hardcoded - no API) ---
    "weather"      : "I don't have live weather access, but I'd say it's always a great day to code! ",
    "how's the weather" : "I can't check live weather, but stay hydrated and keep coding! ",

    # --- Farewells ---
    "bye"          : "Goodbye! It was nice chatting with you. See you next time! ",
    "goodbye"      : "Take care! Come back anytime. ",
    "see you"      : "See you later! Happy coding! ",
    "see ya"       : "See ya! Have a great day!",
    "good night"   : "Good night! Sweet dreams and happy coding tomorrow. ",

    # --- Misc ---
    "thanks"       : "You're welcome! ",
    "thank you"    : "Happy to help! Let me know if you need anything else.",
    "ok"           : "Alright! Is there anything else I can help with?",
    "okay"         : "Okay! Let me know if you need anything.",
    "cool"         : "Glad you think so! ",
}


# -------------------------------------------------------
# PHASE 1: INPUT SANITIZATION FUNCTION
# Cleans the raw user input so our matching works properly.
# .lower()  → converts everything to lowercase
# .strip()  → removes leading and trailing whitespace
# This means "  Hello ", "HELLO", and "hello" all match!
# -------------------------------------------------------
def sanitize_input(raw_input):
    clean = raw_input.lower().strip()
    return clean


# -------------------------------------------------------
# PROCESS INPUT FUNCTION
# Looks up the cleaned input in the knowledge base.
# Uses .get() with a default fallback message - this is
# the professional approach vs a messy if-elif ladder.
# Also handles special tags like __TIME__ and __DATE__.
# -------------------------------------------------------
def process_input(clean_input):
    import datetime  # import here so it's only loaded when needed

    # First check if user wants to exit
    exit_commands = ["exit", "quit", "stop", "close", "end"]
    if clean_input in exit_commands:
        return "__EXIT__"

    # Look up the response in our knowledge base dictionary
    reply = responses.get(clean_input, None)

    # Handle special dynamic responses
    if reply == "__TIME__":
        now = datetime.datetime.now()
        return "The current time is: " + now.strftime("%H:%M:%S")

    if reply == "__DATE__":
        today = datetime.date.today()
        return "Today's date is: " + today.strftime("%A, %B %d, %Y")

    # If no exact match found, try partial keyword matching
    if reply is None:
        reply = partial_match(clean_input)

    # If still nothing found, return the fallback default
    if reply is None:
        reply = "Hmm, I don't quite understand that. Try asking something else, or type 'help'."

    return reply


# -------------------------------------------------------
# PARTIAL MATCH FUNCTION
# If no exact key is found, scan the input for known keywords.
# Example: "can you tell me a joke please?" → matches "joke"
# This makes the bot feel a little smarter without ML!
# -------------------------------------------------------
def partial_match(clean_input):
    # List of keywords to scan for inside the user's message
    keywords = [
        "hello", "hi", "hey", "morning", "evening",
        "how are you", "what's up", "whats up",
        "who are you", "what are you", "your name",
        "help", "time", "date", "today",
        "joke", "funny", "laugh",
        "motivate", "inspire",
        "weather",
        "bye", "goodbye", "good night",
        "thanks", "thank you"
    ]

    for word in keywords:
        if word in clean_input:
            return responses.get(word, None)

    return None  # no keyword found either


# -------------------------------------------------------
# DISPLAY WELCOME MESSAGE
# Called once at the start to greet the user.
# -------------------------------------------------------
def show_welcome():
    print("=" * 50)
    print("       Welcome to DecoBot ")
    print("    Rule-Based AI Chatbot | Project 1")
    print("=" * 50)
    print("  Type 'help' to see what I can do.")
    print("  Type 'exit' to quit the chatbot.")
    print("=" * 50)
    print()


# -------------------------------------------------------
# MAIN FUNCTION — THE HEARTBEAT (INFINITE LOOP)
# This is the core of the chatbot.
# The while True loop keeps the bot alive continuously
# until the user sends an exit command (Kill Command).
# -------------------------------------------------------
def main():
    show_welcome()

    while True:
        # --- PHASE 1: Get raw input from user ---
        raw_input = input("You: ")

        # Handle empty input (user just pressed Enter)
        if not raw_input.strip():
            print("DecoBot: Please type something! (or type 'help')\n")
            continue

        # --- PHASE 1 continued: Sanitize the input ---
        clean_input = sanitize_input(raw_input)

        # --- PHASE 2: Process and match the input ---
        response = process_input(clean_input)

        # --- PHASE 3: Output the response ---
        if response == "__EXIT__":
            print("DecoBot: Goodbye! Thanks for chatting. See you next time! ")
            print("=" * 50)
            break  # Kill command — exit the loop cleanly

        print("DecoBot: " + response)
        print()  # blank line for readability


# -------------------------------------------------------
# ENTRY POINT
# Python convention: only run main() if this file is
# executed directly (not imported as a module).
# -------------------------------------------------------
if __name__ == "__main__":
    main()
