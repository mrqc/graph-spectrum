import psycopg2

def getAllAddressesFromStatement(statement, params):
  addresses = []
  connection = psycopg2.connect(user="", password="", host="", port="", database="")
  cursor = connection.cursor()
  cursor.execute(statement, params)
  for row in cursor.fetchall():
    addresses.append(row[0])
  cursor.close()
  connection.close()
  return list(set(addresses))

def getAllIncomeAddressesFromAddress(address):
  addresses = getAllAddressesFromStatement(
      "SELECT oa2.base58check FROM output_addresses AS oa " +
      "INNER JOIN outputs AS o ON oa.output_id = o.id " +
      "INNER JOIN inputs AS i ON o.transaction_id = i.transaction_id " +
      "INNER JOIN transactions AS t2 ON i.previous_tx_hash = t2.hash " +
      "INNER JOIN outputs AS o2 ON t2.id = o2.transaction_id " + 
      "INNER JOIN output_addresses AS oa2 ON o2.id = oa2.output_id " +
      "WHERE oa.base58check = '%s'", (address,))

def getAllOutgoingAddressesFromAddress(address):
  addresses = getAllAddressesFromStatement(
      "SELECT oa2.base58check FROM output_addresses AS oa " +
      "INNER JOIN outputs AS o ON oa.output_id = o.id " + 
      "INNER JOIN transactions AS t ON o.transaction_id = t.id " +
      "INNER JOIN inputs AS i ON t.hash = i.previous_tx_hash " + 
      "INNER JOIN outputs AS o2 ON i.transaction_id = o2.transaction_id " +
      "INNER JOIN output_addresses AS oa2 ON o2.id = oa2.output_id "+
      "WHERE oa.base58check = '%s'", (address,))

def processAddress(address, level):
  global processedAddresses
  global addressToIndex, indexToAddress
  global adjMat_dir, adjMat_undir
  global D, D_in, D_out
  incomingAddresses = []
  outgoingAddresses = []
  if address not in processedAddresses:
    incomingAddresses = getAllIncomeAddressesFromAddress(address)
    outgoingAddresses = getAllOutgoingAddressesFromAddress(address)

    newIndex = len(adjMat_dir) # also possible: len(adjMat_undir)
    addressToIndex[newIndex] = address
    indexToAddress[address] = newIndex
    adjMat_dir.append([])
    adjMat_undir.append([])
    D.append([])
    D_in.append([])
    D_out.append([])
    edges[address]["in"] = incomingAddresses
    edges[address]["out"] = outgoingAddresses
    processedAddresses.append(address)

    if level < 2:
      for incomingAddress in incomingAddresses:
        processAddress(incomingAddress, level + 1)
      for outgoingAddress in outgoingAddresses:
        processAddress(outgoingAddress, level + 1)

processedAddresses = []
addressToIndex = {}
indexToAddress = {}
edges{}

adjMat_dir = []
adjMat_undir = []
D = []
D_in = []
D_out = []

startAddress = "........."
processAddress(startAddress, 1)

def setAdjMat(from, to, val, adjMat):
  global addressToIndex
  global indexToAddress
  adjMat[addressToIndex[from]][addressToIndex[to]] = val

def getAdjMat(from, to, adjMat):
  global addressToIndex
  global indexToAddress
  return adjMat[addressToIndex[from]][addressToIndex[to]]

for address in addressToIndex:
  adjMat_dir[addressToIndex[address]] = [0] * len(adjMat_dir)
  adjMat_undir[addressToIndex[address]] = [0] * len(adjMat_undir)
  D[addressToIndex[address]] = [0] * len(D)
  D_out[addressToIndex[address]] = [0] * len(D_out)
  D_in[addressToIndex[address]] = [0] * len(D_in)

for address in edges:
  for inAddress in edges[address]["in"]:
    setAdjMat(inAddress, address, 1, adjMat_dir)
  D_in[addressToIndex[address]][addressToIndex[address]] = len(edges[address]["in"])

  for outAddress in edges[address]["out"]:
    setAdjMat(address, outAddress, 1, adjMat_dir)
  D_out[addressToIndex[address]][addressToIndex[address]] = len(edges[address]["out"])

  connectedAddresses = list(set(incomingAddresses + outgoingAddresses))
  for connectedAddress in connectedAddresses:
    setAdjMat(address, connectedAddress, 1, adjMat_undir)
    setAdjMat(connectedAddress, address, 1, adjMat_undir)
  D[addressToIndex[address]][addressToIndex[address]] = len(connectedAddresses)
