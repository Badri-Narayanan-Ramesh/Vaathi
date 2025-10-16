import pyttsx3
print("Init...")
eng = pyttsx3.init()
eng.setProperty("rate", 180)
print("Saying test...")
eng.say("Set the gain to 3.5 decibels at two kilohertz.")
eng.runAndWait()
print("Done.")