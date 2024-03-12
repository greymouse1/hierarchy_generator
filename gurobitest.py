import gurobipy

# Create a Gurobi model
model = gurobipy.Model("Test")

# Add variables
x = model.addVar(name="x", vtype=gurobipy.GRB.CONTINUOUS)
y = model.addVar(name="y", vtype=gurobipy.GRB.CONTINUOUS)

# Set objective function
model.setObjective(x + y, sense=gurobipy.GRB.MAXIMIZE)

# Add constraints
model.addConstr(x + 2 * y <= 10, name="c1")
model.addConstr(2 * x + y <= 10, name="c2")

# Optimize the model
model.optimize()

# Check optimization status
if model.status == gurobipy.GRB.OPTIMAL:
    print("Optimal solution found!")
    print("Objective value:", model.objVal)
    print("Variable values:")
    for var in model.getVars():
        print(var.varName, "=", var.x)
else:
    print("Optimization problem is infeasible or unbounded.")



