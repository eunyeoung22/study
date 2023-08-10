const arr1 = new Array();
const arr2 = [1, 3];

//2. Index position
const fruits = ["사과", "바나나"];
console.log(fruits);
console.log(fruits[0]);
console.log(fruits[1]);

//3. Looping over an Array
for (let i = 0; i < fruits.length; i++) {
    console.log(fruits[i]);
}

for (let fruit of fruits) {
    console.log(fruit);
}

fruits.forEach((fruit) => console.log(fruit));

//4. Addtio, deletion, copy
fruits.push("Strawberry", "peatch");
console.log(fruits);

//마이너스 개념
fruits.pop();
fruits.pop();
console.log(fruits);

//unshift: add an item to the benigging
fruits.unshift("배");
console.log(fruits);

//shift: remove an item from the benigging
fruits.shift();
fruits.shift();
console.log(fruits);

//note!! shift, unshift are slower than pop, push
//pop, push is more faster

//splice : remove an item by index posion
fruits.push("무화과", "레몬", "토마토", "청사과");
console.log(fruits);
fruits.splice(1, 1, "수박", "복숭아");
console.log(fruits);
