from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def create_pdf_file(clusters, k, n):

    with PdfPages(r'C:\Users\User\PycharmProjects\final_project\final_project\Charts.pdf') as export_pdf:
        plt.scatter(df1['Unemployment_Rate'], df1['Stock_Index_Price'], color='green')
        plt.title('Unemployment Rate Vs Stock Index Price', fontsize=10)
        plt.xlabel('Unemployment Rate', fontsize=8)
        plt.ylabel('Stock Index Price', fontsize=8)
        plt.grid(True)
        export_pdf.savefig()
        plt.close()

        plt.plot(df2['Year'], df2['Unemployment_Rate'], color='red', marker='o')
        plt.title('Unemployment Rate Vs Year', fontsize=10)
        plt.xlabel('Year', fontsize=8)
        plt.ylabel('Unemployment Rate', fontsize=8)
        plt.grid(True)
        export_pdf.savefig()
        plt.close()

