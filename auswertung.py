import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os

def parse_list_string(s):
    """Hilfsfunktion: Wandelt Strings in Listen um und fängt 'nan' / 'inf' ab."""
    if pd.isna(s):
        return []
    
    # Ersetzt 'nan' und 'inf' durch 'None', da ast.literal_eval nur das versteht.
    s = str(s).replace('nan', 'None').replace('inf', 'None').replace('NaN', 'None')
    
    try:
        return ast.literal_eval(s)
    except Exception as e:
        print(f"⚠️ Konnte diesen Wert nicht parsen: {s}")
        return []

def load_and_prepare_data(filepath):
    """Lädt die CSV und bereitet die Daten vor."""
    print(f"Lese Daten aus: {filepath}")
    df = pd.read_csv(filepath)
    
    # Wir nutzen jetzt unsere neue sichere Funktion statt direkt ast.literal_eval
    df['k_Verlauf'] = df['k_Verlauf'].apply(parse_list_string)
    df['Kosten_Verlauf'] = df['Kosten_Verlauf'].apply(parse_list_string)
    
    return df

def plot_overall_cost_boxplot(df):
    """Generiert einen Boxplot der finalen Kosten im Vergleich aller Messobjekte."""
    plt.figure(figsize=(10, 6))
    
    # Seaborn Boxplot (zeigt Median, Quartile und Ausreißer)
    sns.boxplot(data=df, x='Messobjekt', y='Finale_Kosten_J', palette='Set2')
    
    plt.title('Vergleich der finalen Optimierungskosten', fontsize=14, fontweight='bold')
    plt.xlabel('Messobjekt', fontsize=12)
    plt.ylabel('Finale Gesamtkosten (J_total)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('plot_1_kosten_boxplot.png', dpi=300)
    print("-> Gespeichert: plot_1_kosten_boxplot.png")

def plot_k_distribution(df):
    """Generiert ein gestapeltes Balkendiagramm für die finale Cluster-Verteilung."""
    # Berechnet die prozentuale Verteilung von k pro Messobjekt
    cross_tab = pd.crosstab(df['Messobjekt'], df['Gewaehltes_k'], normalize='index') * 100
    
    # Plotten
    ax = cross_tab.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    
    plt.title('Verteilung der finalen Cluster-Anzahl (k)', fontsize=14, fontweight='bold')
    plt.xlabel('Messobjekt', fontsize=12)
    plt.ylabel('Häufigkeit in %', fontsize=12)
    plt.legend(title='Gewähltes k', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('plot_2_cluster_verteilung.png', dpi=300)
    print("-> Gespeichert: plot_2_cluster_verteilung.png")

def plot_cost_convergence(df):
    """Generiert die 'U-Form' (Kosten über k) für alle Objekte in einem 2x2 Grid."""
    df_exploded = df[['Messobjekt', 'Run_ID', 'k_Verlauf', 'Kosten_Verlauf']].explode(['k_Verlauf', 'Kosten_Verlauf'])
    
    # Datentypen sicherstellen - NaN Werte (die wir zu None gemacht haben) ignorieren wir hier
    df_exploded = df_exploded.dropna(subset=['k_Verlauf', 'Kosten_Verlauf'])
    df_exploded['k_Verlauf'] = df_exploded['k_Verlauf'].astype(int)
    df_exploded['Kosten_Verlauf'] = df_exploded['Kosten_Verlauf'].astype(float)

    g = sns.relplot(
        data=df_exploded, 
        x='k_Verlauf', 
        y='Kosten_Verlauf', 
        col='Messobjekt', 
        col_wrap=2,          
        kind='line',         
        errorbar='sd',       
        marker='o',          
        height=4, 
        aspect=1.2
    )

    g.fig.suptitle('Kostenverlauf über Cluster-Anzahl (mit Standardabweichung)', fontsize=16, fontweight='bold', y=1.05)
    g.set_axis_labels('Cluster-Anzahl (k)', 'Gesamtkosten (J)')
    
    plt.savefig('plot_3_kosten_verlauf.png', dpi=300, bbox_inches='tight')
    print("-> Gespeichert: plot_3_kosten_verlauf.png")

if __name__ == '__main__':
    csv_file = os.path.expanduser('~/map_ws/pso_evaluation_results.csv')
    
    if not os.path.isfile(csv_file):
        print(f"Fehler: Datei {csv_file} nicht gefunden!")
    else:
        sns.set_theme(style="whitegrid")
        df_results = load_and_prepare_data(csv_file)
        
        print("\nGeneriere Diagramme...")
        plot_overall_cost_boxplot(df_results)
        plot_k_distribution(df_results)
        plot_cost_convergence(df_results)
        
        print("\nAlle Diagramme erfolgreich erstellt! Du findest die Bilder im Ordner map_ws.")