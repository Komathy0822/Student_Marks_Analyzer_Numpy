#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import dataset and library 
import numpy as np 
import matplotlib.pyplot as plt
data = np.genfromtxt(r"C:\Users\ADMIN\Desktop\data set\data_science_student_marks.csv", delimiter=',', dtype=str, skip_header=1)


# In[2]:


# Separate columns
student_id= data[:, 0]
location= data[:, 1]
age= data[:, 2].astype(int)
marks = data[:, 3:].astype(float)
subjects = np.array(['SQL', 'Excel', 'Python', 'PowerBI', 'English'])


# In[3]:


#avg marks 
subject_avg = np.mean(marks, axis=0)
print("Average Marks per Subject:")
for sub, avg in zip(subjects, subject_avg):
    print(f"{sub}: {avg:.2f}")


# In[4]:


total_marks = np.sum(marks, axis=1)
average_marks = np.mean(marks, axis=1)
top5_idx = np.argsort(total_marks)[-5:][::-1]
print("\nTop 5 Students:")
for i in top5_idx:
    print(f"ID: {student_id[i]}, Location: {location[i]}, Total: {total_marks[i]:.1f}")


# In[5]:


unique_locs = np.unique(location)
print("\nAverage Marks by City:")
for city in unique_locs:
    city_mask = (location == city)
    city_avg = np.mean(marks[city_mask], axis=0)
    print(f"{city}: {np.mean(city_avg):.2f}")


# In[6]:


consistency_score = np.std(marks, axis=1)
subject_avg = np.mean(marks, axis=0)
subject_max = np.max(marks, axis=0)
subject_min = np.min(marks, axis=0)
corr_python_sql = np.corrcoef(marks[:, 2], marks[:, 0])[0, 1]
corr_english_excel = np.corrcoef(marks[:, 4], marks[:, 1])[0, 1]


# In[12]:


from datetime import datetime
import os

# 🧠 Assuming your NumPy data variables already exist:
# subjects, marks, average_marks, unique_locs, city_avgs

# ------------------------------------------------------------
# 📂 Create Folders to Store Outputs
# ------------------------------------------------------------
os.makedirs("visuals", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ------------------------------------------------------------
# 🎯 GRAPH 1 — Histogram of Student Average Marks
# ------------------------------------------------------------
plt.figure(figsize=(8,5))
plt.hist(average_marks, bins=10, color='mediumseagreen', edgecolor='black')
plt.title("Distribution of Student Average Marks")
plt.xlabel("Average Marks")
plt.ylabel("Number of Students")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
hist_path = "visuals/student_distribution.png"
plt.savefig(hist_path)
plt.close()

# ------------------------------------------------------------
# 🎯 GRAPH 2 — Correlation Heatmap
# ------------------------------------------------------------
corr_matrix = np.corrcoef(marks.T)
plt.figure(figsize=(6,5))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label="Correlation Coefficient")
plt.xticks(range(len(subjects)), subjects, rotation=45)
plt.yticks(range(len(subjects)), subjects)
plt.title("Subject Correlation Heatmap")
plt.tight_layout()
heatmap_path = "visuals/subject_correlation.png"
plt.savefig(heatmap_path)
plt.close()

# ------------------------------------------------------------
# 🧾 FINAL REPORT GENERATION (with Visual Descriptions)
# ------------------------------------------------------------
report = []
report.append("🎓 STUDENT PERFORMANCE ANALYSIS REPORT\n")
report.append("="*60 + "\n\n")

# Section 1 — Subject-wise Summary
report.append("📊 SUBJECT-WISE PERFORMANCE SUMMARY:\n")
for sub, avg, mx, mn in zip(subjects, np.mean(marks, axis=0), np.max(marks, axis=0), np.min(marks, axis=0)):
    report.append(f"{sub:<10} | Avg: {avg:6.2f} | Highest: {mx:5.1f} | Lowest: {mn:5.1f}\n")
report.append("\n")

# Section 2 — Overall Performance
overall_avg = np.mean(average_marks)
overall_max = np.max(average_marks)
overall_min = np.min(average_marks)

report.append("🏆 OVERALL PERFORMANCE:\n")
report.append(f"Average Marks of All Students : {overall_avg:.2f}\n")
report.append(f"Highest Average Marks         : {overall_max:.2f}\n")
report.append(f"Lowest Average Marks          : {overall_min:.2f}\n\n")

# Section 3 — City-wise
report.append("📍 CITY-WISE PERFORMANCE:\n")
for city, avg in zip(unique_locs, city_avg):
    report.append(f"{city:<15} -> Average Marks: {avg:.2f}\n")
report.append("\n")

# Section 4 — Correlation Summary
report.append("🤝 SUBJECT CORRELATION (Simplified View):\n")
for i in range(len(subjects)):
    for j in range(i+1, len(subjects)):
        corr = np.corrcoef(marks[:, i], marks[:, j])[0, 1]
        report.append(f"{subjects[i]} vs {subjects[j]} : {corr:.2f}\n")
report.append("\n")

# Section 5 — Visualizations (inline in report)
report.append("🖼️ VISUAL INSIGHTS:\n")
report.append("------------------------------------------------------------\n")
report.append("Figure 1: Distribution of Student Average Marks\n")
report.append("File: visuals/student_distribution.png\n")
report.append("Description: This histogram shows how students’ average marks are distributed.\n")
report.append("Peaks indicate the most common performance range.\n\n")

report.append("Figure 2: Subject Correlation Heatmap\n")
report.append("File: visuals/subject_correlation.png\n")
report.append("Description: Displays correlation between subjects.\n")
report.append("Brighter cells show stronger positive relationships.\n\n")
report.append("------------------------------------------------------------\n")

# Section 6 — Automated Summary Conclusion
best_city = unique_locs[np.argmax(city_avg)]
best_subject = subjects[np.argmax(np.mean(marks, axis=0))]
strongest_pair = None
strongest_corr = -1
for i in range(len(subjects)):
    for j in range(i+1, len(subjects)):
        corr = np.corrcoef(marks[:, i], marks[:, j])[0, 1]
        if corr > strongest_corr:
            strongest_corr = corr
            strongest_pair = (subjects[i], subjects[j])

report.append("SUMMARY CONCLUSION:\n")
report.append(f"->Students from {best_city} performed the best overall.\n")
report.append(f"->The strongest correlation was between {strongest_pair[0]} and {strongest_pair[1]} ({strongest_corr:.2f}).\n")
report.append(f"->The top-performing subject on average was {best_subject}.\n")
report.append("\n")

# Footer
report.append("="*60 + "\n")
report.append(f"📅 Report Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Save report
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"reports/Student_Performance_Report_{timestamp}.txt"
with open(filename, "w", encoding="utf-8") as f:
    f.writelines(report)

print(f"📁 Report saved successfully as '{filename}'")
print("📊 Graphs saved inside 'visuals/' folder.")


# In[ ]:




