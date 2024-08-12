FROM python:3.10-slim-bullseye

# Set the working directory
WORKDIR /usr/src/app

# Copy necessary files
COPY main.py /usr/src/app/main.py
COPY master_emb_list.pkl /usr/src/app/master_emb_list.pkl
COPY master_skills_list.txt /usr/src/app/master_skills_list.txt

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the helper script and give it execution permissions
COPY helper_script.sh /usr/src/app/helper_script.sh
RUN chmod +x helper_script.sh

# Set the PORT environment variable
ENV PORT=8080

# Expose the port the app runs on
EXPOSE 8080

# Start the application
CMD ["./helper_script.sh"]
