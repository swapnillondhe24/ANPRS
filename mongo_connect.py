import pymongo

# Replace the placeholders with your connection string and database name
connection_string = "mongodb+srv://Admin:Admin@anprs.csdqtst.mongodb.net/?retryWrites=true&w=majority"
# connection_string = "mongodb+srv://root:phts16cVFjrXAeeI@decends.y0zu65e.mongodb.net/"
db_name = "ANPRS"
print("Connecting to the database...")
# Create a MongoClient object with the connection string
client = pymongo.MongoClient(connection_string)

print("Connected to the database successfully.")
# Access the database using the client and database name
db = client[db_name]
print("The following databases are available", client.list_database_names())

# Now you can perform operations on the database, such as finding documents in a collection
collection = db["LicensePlates"]
# print("The following collections are available", db.list_collection_names())
documents = collection.find({"LP_number": "MH12LJ3236"})
# print("The following documents are available in the collection:", collection.name)
texts_filtered = 'MH12LJ3236'
name = find_document(texts_filtered)
# print(documents)
for i in documents:
    print(i['Name'],"Owns",i['LP_number'])
