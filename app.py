import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import json
from  PIL import Image
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from geopy.distance import geodesic
import networkx as nx


img = Image.open('img.png')

def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)


st.set_page_config(page_title="Supply Chain Optimization",page_icon=img, layout="wide")

video_html = """
		<style>

		#myVideo {
		  position: fixed;
		  right: 0;
		  bottom: 0;
		  min-width: 100%; 
		  min-height: 100%;
		}

		.content {
		  position: fixed;
		  bottom: 0;
		  background: rgba(0, 0, 0, 0.5);
		  color: #f1f1f1;
		  width: 100%;
		  padding: 20px;
		}

		</style>	
		<video autoplay muted loop id="myVideo">
		  <source src="")>
		  Your browser does not support HTML5 video.
		</video>
        """

st.markdown(video_html, unsafe_allow_html=True)
# Load your dataset
# Assuming your dataset is in a CSV file named 'supply_chain_data.csv'
df = pd.read_csv('supply_chain_data.csv')

# Title and description

# Sidebar with three tabs
with st.sidebar:
    selected = option_menu(
        menu_title="Analytic Tool",
        options=["Home", "Route Recommendation", "Cost Minimization", "Environmental Impact", "Product Availability"],
        icons=["house", "activity", "clipboard-data", "graph-up-arrow"],
        menu_icon=None,
        styles={
            "container": {"padding": "0!important", "background-color": "#DBF0F9"},
            "icon": {"color": "black", "font-size": "20px"},
            "nav-link": {
                "font-size": "20px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#9BBEC8",
            },
            "nav-link-selected": {"background-color": "#427D9D"},
        },
    )

if selected == "Route Recommendation":
    def get_coordinates(location):
        geolocator = Nominatim(user_agent="geo_locator")
        try:
            coordinates = geolocator.geocode(location)
            return (coordinates.latitude, coordinates.longitude)
        except (AttributeError, GeocoderTimedOut):
            st.warning("Location not found. Please enter a valid location.")
            return None

    def calculate_distance(location_from, location_to):
    # Calculate distance between two locations using geopy
        distance = geodesic(location_from, location_to).kilometers
        return distance

    def create_map_data(locations):
    # Create a DataFrame with columns 'LAT', 'LON', and 'DISTANCE'
        map_data = pd.DataFrame({
        'LAT': [location[0] for location in locations],
        'LON': [location[1] for location in locations]
        })
        return map_data

    def create_graph(locations):
    # Create a graph using networkx
        G = nx.Graph()

    # Add nodes to the graph
        for i, location in enumerate(locations):
            G.add_node(i, pos=location)

    # Add edges (connections between nodes) with distances as weights
        for i in range(len(locations) - 1):
            for j in range(i + 1, len(locations)):
                distance = calculate_distance(locations[i], locations[j])
                G.add_edge(i, j, weight=distance)

        return G

    def find_paths(G, start, end, num_paths=3):
    # Find all simple paths between start and end nodes
        paths = list(nx.all_simple_paths(G, source=start, target=end))

    # Sort paths based on total distance
        sorted_paths = sorted(paths, key=lambda path: sum(G[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1)))

    # Return at most num_paths shortest paths
        return sorted_paths[:num_paths]

    def recommend_vehicle(weight):
    # Recommend a vehicle based on the weight
        if weight <= 50:
            return 'Bike'
        elif weight <= 200:
            return 'Scooter'
        elif weight <= 500:
            return 'Sedan'
        elif weight <= 1000:
            return 'Van'
        elif weight <= 5000:
            return 'Truck'
        elif weight <= 10000:
            return 'Lorry'
        elif weight <= 20000:
            return 'Ship'
        elif weight <= 50000:
            return 'Airplane'
        else:
            return 'Weight exceeds limit'

    def transportation_mode(weight):
    # Determine the mode of transportation based on the weight
        if weight <= 5000:
            return 'Road'
        elif weight <= 20000:
            return 'Ship'
        elif weight <= 50000:
            return 'Airplane'
        else:
            return 'Transportation mode not determined'

    def show_route_recommendation_tab():
        st.title("Multi-Objective Supply Chain Optimization - Route Recommendation Tab")

    # Collect user input for starting and ending locations
        from_location_name = st.text_input("Enter From Location (City or State):")
        to_location_name = st.text_input("Enter To Location (City or State):")
        weight = st.number_input("Enter Weight of Loads/Products (in kg):", min_value=1, max_value=50000)

        if st.button("Run"):
        # Get coordinates for the user input locations
            location_from = get_coordinates(from_location_name) if from_location_name else None
            location_to = get_coordinates(to_location_name) if to_location_name else None

            if location_from and location_to:
                locations = [location_from, location_to]

            # Create map data for displaying markers on the map
                map_data = create_map_data(locations)
                st.map(map_data)

            # Create a graph based on the locations
                G = create_graph(locations)

            # Find at most three shortest paths
                paths = find_paths(G, start=0, end=1, num_paths=3)

            # Display the recommended paths, distances, vehicles, and transportation mode
                st.header("Recommended Paths:")
                for i, path in enumerate(paths, start=1):
                    distance = sum(G[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))
                    recommended_vehicle = recommend_vehicle(weight)
                    mode_of_transportation = transportation_mode(weight)

                    st.write(f"Best Route: {' -> '.join(map(str, path[1:]))}, Distance - {distance:.2f} kilometers")
                    st.write(f"Recommended Vehicle: {recommended_vehicle}")
                    st.write(f"Transportation Mode: {mode_of_transportation}")
                    st.write("Note: Products will be delivered safely.")

    if __name__ == "__main__":
        show_route_recommendation_tab()
if selected == "Cost Minimization":
    st.header("Cost Minimization Analysis")
    # User input inside the window
    st.subheader("User Input")
    product_type = st.selectbox("Select Product Type", df['Product type'].unique())
    max_price = st.number_input("Maximum Price", min_value=0, max_value=int(df['Price'].max()))
    min_availability = st.number_input("Minimum Availability", min_value=0, max_value=int(df['Availability'].max()))

    # Filtering data based on user input
    filtered_data = df[(df['Product type'] == product_type) & (df['Price'] <= max_price) & (df['Availability'] >= min_availability)]

    # Calculate 'Total Costs' as the sum of three cost columns
    filtered_data['Total Costs'] = filtered_data['Manufacturing costs'] + filtered_data['Shipping costs'] + filtered_data['Costs']

    # Cost Minimization Analysis tab content
    st.header("Cost Minimization Analysis")

    # Dropdown to select analysis section
    analysis_section = st.selectbox("Select Analysis Section", ["None","Statistics and Recommendations", "Visualization"])

    if analysis_section == "Statistics and Recommendations":
        # Calculate and display cost statistics
        cost_stats = filtered_data[['Manufacturing costs', 'Shipping costs', 'Costs', 'Total Costs']].describe()
        st.subheader("Cost Statistics")
        st.write(cost_stats)
    
        # Recommendations for Cost Reduction
        st.subheader("Recommendations for Cost Reduction")

        recommendation_count = 1
        for index, row in filtered_data.iterrows():
            st.write(f"{recommendation_count}. Product Type: {row['Product type']}")
            st.write(f"   Total Costs: {row['Total Costs']}")
    
            manufacturing_cost_msg = "   Manufacturing costs are within an acceptable range."
            if row['Manufacturing costs'] > 100:
                manufacturing_cost_msg = "   Manufacturing costs are high and need optimization."
            st.write(manufacturing_cost_msg)
    
            shipping_cost_msg = "   Shipping costs are within an acceptable range."
            if row['Shipping costs'] > 50:
                shipping_cost_msg = "   Shipping costs are high and should be reviewed."
            st.write(shipping_cost_msg)
    
            other_costs_msg = "   Other costs are within an acceptable range."
            if row['Costs'] > 50:
                other_costs_msg = "   Other costs are high and require attention."
            st.write(other_costs_msg)
    
            total_costs_msg = "   The total costs are acceptable (below $200)."
            if row['Total Costs'] > 200:
                total_costs_msg = "   The total costs need reduction (exceeds $200)."
            st.write(total_costs_msg)
    
            recommendation_count += 1
    
    elif analysis_section == "Visualization":
        # Dropdown to select the graph to display
        st.subheader("Select a Graph to Display")
        selected_graph = st.selectbox("Choose a Graph", ["Line Plot", "Histogram", "Scatter Plot", "Pie Chart"])
    
        # Based on the selection, display the corresponding graph
        if selected_graph == "Line Plot":
            st.header("Line Plot")
            fig_line = px.line(filtered_data, x="Product type", y="Manufacturing costs", title="Manufacturing Costs Over Product Types")
            st.plotly_chart(fig_line)
        elif selected_graph == "Histogram":
            st.header("Histogram")
            fig_hist = px.histogram(filtered_data, x="Shipping costs", title="Shipping Costs Distribution")
            st.plotly_chart(fig_hist)
        elif selected_graph == "Scatter Plot":
            st.header("Scatter Plot")
            fig_scatter = px.scatter(filtered_data, x="Costs", y="Total Costs", title="Scatter Plot of Costs vs. Total Costs")
            st.plotly_chart(fig_scatter)
        elif selected_graph == "Pie Chart":
            st.header("Pie Chart")

            # Calculate the number of products needing reduction and the number of products that are acceptable
            num_need_reduction = len(filtered_data[filtered_data['Total Costs'] > 200])
            num_acceptable = len(filtered_data[filtered_data['Total Costs'] <= 200])

            # Create a pie chart to show the distribution
            fig_pie = px.pie(
                names=["Need Reduction", "Acceptable"],
                values=[num_need_reduction, num_acceptable],
                title="Product Cost Status"
            )
            st.plotly_chart(fig_pie)
if selected == "Environmental Impact":
    st.header("Environmental Impact Analysis")
    option = st.selectbox("Select an Analysis:", ["None","Carbon Emission Analysis", "Defect Rates Analysis", "Manufacturing Lead Time Analysis", "Manufacturing Costs Analysis", "Transportation Costs Analysis"])
    if option == "Carbon Emission Analysis":
        st.header("Environmental Impact - Carbon Emissions Analysis")
            # Add statistics calculations and recommendations based on the selected data

        def calculate_emissions(df):
                # Create a dictionary of emission factors for different transportation modes
                emission_factors = {
                    "Air": 800,   # Example: 800 kg CO2 per ton-km for air transport
                    "Sea": 15,    # Example: 15 kg CO2 per ton-km for sea transport
                    "Rail": 5,    # Example: 5 kg CO2 per ton-km for rail transport
                    "Road": 100,  # Example: 100 kg CO2 per ton-km for road transport
                }

                # Create a new column for emissions based on the transportation mode
                df['Emissions'] = df['Transportation modes'].map(emission_factors)

                # Calculate emissions based on shipping times (simplified)
                df['Emissions'] += df['Shipping times']

                return df

            # Calculate emissions
        df = calculate_emissions(df)

            # Output message based on emissions
        st.subheader("Environmental Impact - Carbon Emissions Analysis")

            # Calculate ppm based on emissions
            # You can adjust the conversion factor as needed
        ppm_conversion_factor = 1
        df['ppm'] = df['Emissions'] * ppm_conversion_factor

            # Create a button to select specific data
        if st.button("Select Data by Emission Level"):
                # Select high emissions
                high_emission_data = df[df['ppm'] > 430]  # Adjust the threshold as needed
                # Select data within the acceptable range
                acceptable_emission_data = df[df['ppm'] <= 430]  # Adjust the threshold as needed

                # Display selected data
                st.subheader("Selected Data")
                if not high_emission_data.empty:
                    st.warning("Data with high emissions:")
                    st.dataframe(high_emission_data)
                    st.write("Recommendation: Since the ppm value is high, efforts should be made to reduce carbon emissions.")
                if not acceptable_emission_data.empty:
                    st.info("Data within the acceptable range:")
                    st.dataframe(acceptable_emission_data)

            # Visualization for Environmental Impact (optional)

            # Visualization 1: Line Plot
        st.subheader("Line Plot")
        # Create a line plot using Plotly Express
        fig_line = px.line(df, x='Location', y='Emissions', title="Emissions by Location Over Time")
        st.plotly_chart(fig_line)

            # Visualization 2: Scatter Plot
        st.subheader("Scatter Plot")
            # Create a scatter plot using Plotly Express
        fig_scatter = px.scatter(df, x='Location', y='Emissions', title="Emissions by Location")
        st.plotly_chart(fig_scatter)

            # Visualization 3: Bar Chart
        st.subheader("Bar Chart")
            # Create a bar chart using Plotly Express
        fig_bar = px.bar(df, x='Product type', y='Emissions', title="Emissions by Product Type")
        st.plotly_chart(fig_bar)
    elif option == "Defect Rates Analysis":
        st.header("Defect Rates Analysis")

        # Select a column for grouping (Product type, Supplier name, or Location)
        group_by_column = st.selectbox("Select a Column for Analysis:", ['Product type', 'Supplier name', 'Location'])

        # Group data and calculate defect rates
        defect_rates = df.groupby(group_by_column)['Defect rates'].mean()

        st.subheader(f"Defect Rates by {group_by_column}")
        st.write(defect_rates)

        # Calculate and display the highest defect rate and the corresponding group
        max_defect_rate = defect_rates.max()
        high_defect_group = defect_rates[defect_rates == max_defect_rate].index[0]

        st.subheader("Highest Defect Rate")
        st.write(f"The highest defect rate is {max_defect_rate:.2%} in the group of {high_defect_group}.")

        # Recommendations
        st.subheader("Recommendations")
        st.write("1. Investigate the root causes of defects in the group with the highest defect rate.")
        st.write("2. Implement quality control measures and inspections to reduce defects.")
        st.write("3. Collaborate with suppliers or locations with high defect rates to improve product quality.")
        st.header("Defect Rates Visualization")

    # Create a bar chart showing defect rates by the selected column
        fig_bar = px.bar(df, x=group_by_column, y='Defect rates', title=f"Defect Rates by {group_by_column}")
        st.plotly_chart(fig_bar)
    elif option == "Manufacturing Lead Time Analysis":
        st.header("Manufacturing Lead Time Analysis")

        # Select a column for grouping (Product type, Supplier name, or Location)
        group_by_column = st.selectbox("Select a Column for Analysis:", ['Product type', 'Supplier name', 'Location'])

        # Group data and calculate average manufacturing lead times
        avg_lead_times = df.groupby(group_by_column)['Manufacturing lead time'].mean()

        st.subheader(f"Average Manufacturing Lead Times by {group_by_column}")
        st.write(avg_lead_times)

        # Calculate and display the highest lead time and the corresponding group
        max_lead_time = avg_lead_times.max()
        high_lead_time_group = avg_lead_times[avg_lead_times == max_lead_time].index[0]

        st.subheader("Highest Manufacturing Lead Time")
        st.write(f"The highest manufacturing lead time is {max_lead_time} days in the group of {high_lead_time_group}.")

        # Assess environmental impact (you can define your own calculation)
        # For example, assume that extended lead times result in increased energy consumption
        # You can use an appropriate formula to estimate the environmental impact

        # Calculate energy consumption based on lead times
        avg_energy_consumption = max_lead_time * 10  # Example formula (adjust as needed)

        st.subheader("Estimated Energy Consumption Due to Extended Lead Times")
        st.write(f"The estimated energy consumption is {avg_energy_consumption} kWh.")

        # Recommendations
        st.subheader("Recommendations")
        st.write("1. Investigate the processes or products with long lead times and high environmental impact.")
        st.write("2. Implement strategies to reduce lead times and optimize production schedules.")
        st.write("3. Implement energy-efficient manufacturing practices to conserve energy.")
        st.write("4. Explore opportunities for just-in-time manufacturing to reduce lead times.")

        st.header("Manufacturing Lead Time Visualization")

    # Create a bar chart showing average manufacturing lead times by the selected column
        fig_bar = px.bar(df, x=group_by_column, y='Manufacturing lead time', title=f"Average Manufacturing Lead Times by {group_by_column}")
        st.plotly_chart(fig_bar)
    elif option == "Manufacturing Costs Analysis":
        st.header("Manufacturing Costs Analysis")

        # Select a column for grouping (Product type or Supplier name)
        group_by_column = st.selectbox("Select a Column for Analysis:", ['Product type', 'Supplier name'])

        # Group data and calculate average manufacturing costs
        avg_manufacturing_costs = df.groupby(group_by_column)['Manufacturing costs'].mean()

        st.subheader(f"Average Manufacturing Costs by {group_by_column}")
        st.write(avg_manufacturing_costs)

        # Calculate and display the highest manufacturing costs and the corresponding group
        max_costs = avg_manufacturing_costs.max()
        high_costs_group = avg_manufacturing_costs[avg_manufacturing_costs == max_costs].index[0]

        st.subheader("Highest Manufacturing Costs")
        st.write(f"The highest manufacturing costs are ${max_costs} in the group of {high_costs_group}.")

        # Assess environmental impact (you can define your own calculation)
        # For example, assume that higher production costs result in increased resource consumption
        # You can use an appropriate formula to estimate the environmental impact

        # Calculate resource consumption based on manufacturing costs
        avg_resource_consumption = max_costs * 0.05  # Example formula (adjust as needed)

        st.subheader("Estimated Resource Consumption Due to Higher Manufacturing Costs")
        st.write(f"The estimated resource consumption is {avg_resource_consumption} kg of resources.")

        # Recommendations
        st.subheader("Recommendations")
        st.write("1. Investigate products or suppliers with high manufacturing costs and environmental impact.")
        st.write("2. Explore cost-reduction strategies that can lead to lower environmental impact.")
        st.write("3. Consider resource-efficient manufacturing practices to reduce resource consumption.")
        st.write("4. Evaluate potential supplier partnerships for cost-effective sourcing.")
    elif option == "Transportation Costs Analysis":
        st.header("Transportation Costs Analysis")

    # Additional Analysis: Analyzing Transportation Costs
        st.subheader("Analyzing Transportation Costs")

        # Calculate the sum of transportation costs
        total_costs = df.groupby('Shipping carriers')['Shipping costs'].sum()
        st.write("Total Transportation Costs by Carrier:")
        st.write(total_costs)

        # Calculate the average shipping costs
        average_costs = df['Shipping costs'].mean()
        st.write(f"Average Shipping Costs: ${average_costs:.2f}")

        # Identify the costliest route
        costliest_route = df.loc[df['Shipping costs'].idxmax()]
        st.write("Costliest Route:")
        st.write(costliest_route)
if selected == "Product Availability":
    st.header("Product Availability Analysis")

    # 1. Analyze availability by product type
    average_availability_by_product_type = df.groupby('Product type')['Availability'].mean()
    highest_availability_product_type = average_availability_by_product_type.idxmax()
    lowest_availability_product_type = average_availability_by_product_type.idxmin()

    st.subheader("Analyze availability by product type:")
    st.write("Average Availability by Product Type:")
    st.write(average_availability_by_product_type)
    st.write(f"Product type with the Highest Availability: {highest_availability_product_type}")
    st.write(f"Product type with the Lowest Availability: {lowest_availability_product_type}")

    # 2. Analyze availability by location
    average_availability_by_location = df.groupby('Location')['Availability'].mean()
    highest_availability_location = average_availability_by_location.idxmax()
    lowest_availability_location = average_availability_by_location.idxmin()

    st.subheader("Analyze availability by location:")
    st.write("Average Availability by Location:")
    st.write(average_availability_by_location)
    st.write(f"Location with the Highest Availability: {highest_availability_location}")
    st.write(f"Location with the Lowest Availability: {lowest_availability_location}")

    # 4. Recommend strategies to improve availability (You can add your recommendations)
    st.subheader("Recommendations:")
    st.write("Based on the analysis, consider the following recommendations:")
    st.write("1. Implement better inventory management practices.")
    st.write("2. Diversify suppliers to reduce stockouts.")
    st.write("3. Monitor demand patterns and adjust stocking levels accordingly")

    # 6. Analyze the impact of product type on availability
    correlation_product_type = df.groupby('Product type')['Availability'].corr(df['Availability'], method='spearman')
    st.subheader("Analyze the Impact of Product Type on Availability:")
    st.write("Correlation between Product Type and Availability:")
    st.write(correlation_product_type)

    # 7. Analyze the impact of location on availability
    correlation_location = df.groupby('Location')['Availability'].corr(df['Availability'], method='spearman')
    st.subheader("Analyze the Impact of Location on Availability:")
    st.write("Correlation between Location and Availability:")
    st.write(correlation_location)

        # 9. Analyze the Impact of Lead Time on Availability
    lead_time_availability_correlation = df[['Lead times', 'Availability']].corr()
    st.subheader("Impact of Lead Time on Availability")
    st.write("Correlation between Lead Time and Availability:")
    st.write(lead_time_availability_correlation)

    # 11. Sales Impact of Availability
    df['Estimated Sales Impact'] = (df['Availability'] / 100) * df['Revenue generated']
    total_sales_impact = df['Estimated Sales Impact'].sum()
    st.subheader("Sales Impact of Availability")
    st.write(f"Total Estimated Sales Impact: ${total_sales_impact:,.2f}")
if selected == "Home":
    st.title('Multi-Objective Supply Chain Optimization')
    st.subheader("Analyzing, Optimizing, and Visualizing Your Supply Chain")
    st.write("Welcome to the Multi-Objective Supply Chain Optimization Web app, a powerful tool that empowers you to delve deep into your supply chain operations. With this app, you can perform comprehensive analysis across a range of crucial factors, including cost minimization, environmental impact reduction, and product availability. By leveraging data-driven insights, you can make informed decisions to optimize your supply chain while balancing multiple objectives.")
    with st.container():
        left_column, right_column = st.columns(2)  
        with left_column:
            st.header("Supply Chain")
            st.write("A supply chain is a complex network of organizations, people, activities, information, and resources involved in the creation and delivery of a product or service to consumers. It encompasses every step in the production process, from the extraction of raw materials to the distribution of the final product. Effective supply chain management is crucial for businesses to ensure that products or services are delivered to customers efficiently, on time, and at the right cost. ")
        with right_column:
            lottie_supply = load_lottiefile("LottieFiles/Animation - 1699190019496.json")
            st_lottie(
                lottie_supply,
                height=300,
                width=300,
            )
        with left_column:
            lottie_supply = load_lottiefile("LottieFiles/Supply.json")
            st_lottie(
                lottie_supply,
                height=400,
                width=400,
            )
        with right_column:
            st.header("Website Aims to.....")
            st.write("Our Project Multi-Objective Supply Chain Optimization aims to enhance supply chain management by simultaneously optimizing various objectives. This includes Recommending the routes and Vehicles for the users, minimizing costs, reducing environmental impact and ensuring product availability. By leveraging data analysis and predictive modeling, the project seeks to provide actionable insights for decision-makers, helping them make informed choices to improve efficiency, sustainability, and customer satisfaction within the supply chain.")
        with left_column:
            st.header("Route Recommendation Analysis")
            st.write("Discover efficient supply chain logistics with our 'Route Recommendation' tab! Input your locations and load weight to get up to three optimal routes, recommended vehicles, and transportation modes. Visualize it on an interactive map for informed decision-making. Welcome to smarter supply chain management!")
        with right_column:
            lottie_supply = load_lottiefile("LottieFiles/Route.json")
            st_lottie(
                lottie_supply,
                height=400,
                width=400,
            )
        with right_column:
            st.header("Cost Minimization Analysis")
            st.write("In Cost Minimization Analysis, our project focuses on cutting supply chain costs. It involves optimizing transportation, inventory, and procurement expenses to boost profitability. We use data-driven insights and predictive models to help businesses save money, streamline operations, and make informed decisions. Our goal is to enhance financial efficiency and maintain a balance with other supply chain goals like sustainability and product availability, ultimately helping businesses thrive in a competitive landscape.")
        with left_column:
            lottie_supply = load_lottiefile("LottieFiles/Cost.json")
            st_lottie(
                lottie_supply,
                height=400,
                width=400,
            )
        with right_column:
            lottie_supply = load_lottiefile("LottieFiles/Environment.json")
            st_lottie(
                lottie_supply,
                height=400,
                width=400,
            )
        with left_column:
            st.header("Environmental Impact Analysis")
            st.write("In Environmental Impact Analysis, our project centers on assessing the ecological footprint of supply chain operations. We examine factors such as transportation modes, packaging materials, and energy consumption to quantify carbon emissions and environmental impacts. By implementing sustainable practices and adopting eco-friendly solutions, our project aims to reduce the environmental burden of supply chains. We empower businesses to make environmentally responsible decisions, fostering a greener and more sustainable future.")
        with right_column:
            st.header("Product Availability Analysis")
            st.write("Product Availability Analysis is a pivotal component of our project, focusing on ensuring product availability in the supply chain. We delve into variables like stockouts, lead times, and demand patterns to optimize stock levels and enhance product accessibility. By leveraging data-driven insights, we enable businesses to meet customer demands efficiently while minimizing inventory costs. Our analysis improves the overall customer experience and the supply chain's profitability, ensuring products are available when and where needed.")
        with left_column:
            lottie_supply = load_lottiefile("LottieFiles/Product.json")
            st_lottie(
                lottie_supply,
                height=400,
                width=400,
            )