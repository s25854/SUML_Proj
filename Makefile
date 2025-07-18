install:
	pip install --upgrade pip && \
	pip install -r requirements.txt && \
	pip install black pytest papermill cml

format:
	black app/ model/

train:
	papermill training.ipynb output.ipynb

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md

	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./Results/model_results.png)' >> report.md

	cml comment create report.md


build:
	docker build -t wine-streamlit-app .

run:
	streamlit run app/streamlit_app.py

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new results"
	git push --force origin HEAD:update
