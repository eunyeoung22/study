const name = "ellie";
const age = 4;
print(name, age);
function print(name, age) {
    console.log(name);
}

//3. Property value shorthand

const user = { name: "ellie", age: "20" };
const user2 = user;
console.log(user);

//old way
const user3 = {};
for (key in user) {
    user3[key] = user[key];
}
console.log(user3);

const user4 = Object.assign({}, user3);
console.log(user4);
