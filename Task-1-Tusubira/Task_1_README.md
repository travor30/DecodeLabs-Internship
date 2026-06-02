#  Task 1 — Rule-Based AI Chatbot

**DecodeLabs Industrial Training | Batch 2026 | Artificial Intelligence**


##  Project Overview

A conversational AI chatbot built entirely from **control flow and logic** — no machine learning, no external APIs. The bot runs in a continuous loop, accepts user input, sanitizes it, matches it against a knowledge base dictionary, and returns intelligent responses. This project demonstrates the foundational architecture that underpins all modern AI guardrail systems.

> *"Before you build systems that learn on their own, you must master the art of teaching a machine through explicit if-else instructions."*
> — DecodeLabs Project Brief


## Objectives
Understand the difference between **deterministic (rule-based)** and **probabilistic (ML-based)** AI
Build a working chatbot using **pure Python logic** — no libraries required
Implement the **IPO Model** (Input → Process → Output) as an architectural framework
Apply **O(1) dictionary lookup** instead of O(n) if-elif chains

##  Architecture


                 INFINITE LOOP (while True)          
                                                       
  INPUT          PROCESS           OUTPUT             
              
 raw_input  →   sanitize()    →   print response    
                ↓                                     
                 exact match?  →   dictionary.get()  
                                    
                 keyword scan? →   partial_match()   
                ↓                                     
                 fallback      →   default reply    
                 ↓                                     
                 exit command? →   break (kill cmd)  



## Features

 Feature                               Description 

 **30+ responses**                Greetings, identity, time/date, jokes, motivation, weather, farewells |
 **Input sanitization**         `.lower().strip()` normalizes all case and whitespace variations |
 **Two-layer matching**           Exact dictionary lookup first, keyword scan as fallback |
 **Dynamic time & date**        Live responses using Python's built-in `datetime` module |
 **Multiple exit words**        `exit`, `quit`, `stop`, `close`, `end` all terminate cleanly |
 **Empty input guard**           Prevents crashes on accidental Enter press |
 **Fallback response**          Never crashes on unknown input — always replies gracefully |


##  Key Concepts Demonstrated

### Dictionary vs If-Elif — Performance Comparison

 Approach | Complexity | Notes |

 `if-elif` ladder | O(n) | Checks every condition linearly — slows as rules grow |
 **Dictionary `.get()`** | **O(1)** | Hash-based instant lookup — same speed at 5 or 5000 rules |

### The White Box Principle
Rule-based systems are fully **traceable**: Input → Logic → Output. No mystery, no hallucination risk. Critical in Finance, Healthcare, and compliance systems.



##  File Structure


Task-1-Rule-Based-Chatbot/

 chatbot.py          Main chatbot script (run this)
 README.md           This file


##  How to Run

**Requirements:** Python 3.6+ | No external libraries needed

bash
python chatbot.py


**Example Session:**

==================================================
       Welcome to DecoBot 
    Rule-Based AI Chatbot | Project 1
==================================================

You: hello
DecoBot: Hey there! I'm DecoBot. How can I help you today?

You: what time is it
DecoBot: The current time is: 14:32:07

You: tell me a joke
DecoBot: Why do programmers prefer dark mode? Because light attracts bugs! 

You: exit
DecoBot: Goodbye! Thanks for chatting. See you next time! 










