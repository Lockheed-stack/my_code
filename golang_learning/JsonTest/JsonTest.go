package JsonTest

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
)

type Person struct {
	Name    string
	Age     int
	Email   string
	Parents []string
}

func Marshal_json() {
	var p = Person{
		Name:  "Alex",
		Age:   19,
		Email: "alex@gmail.com",
	}
	b, err := json.Marshal(p)
	if err != nil {
		log.Fatal("error happend in json Marshal")
	}
	fmt.Printf("b: %v\n", string(b))
}
func UnMarshal_json() {
	b := []byte(`{"Name":"Alex","Age":19,"Email":"alex@gmail.com"}`)
	var p Person
	json.Unmarshal(b, &p)
	fmt.Printf("p: %v\n", p)
}

func Nested_type_UnMarshal() {
	b := []byte(`{"Name":"Alex","Age":19,"Email":"alex@gmail.com","Parents":["jack","Jesus"]}`)
	var any_test any
	json.Unmarshal(b, &any_test)
	fmt.Printf("any_test type: %T, value:%v\n", any_test, any_test)
	if _, ok := any_test.(map[string]any); ok {
		for key, value := range any_test.(map[string]interface{}) {
			fmt.Printf("key:%v, value:%v\n", key, value)
		}
	}

}

func Json_read_file() {
	f, err := os.OpenFile("test.json", os.O_RDONLY, 0644)
	if err != nil {
		log.Panic("no such file or directory")
	}
	defer f.Close()

	d := json.NewDecoder(f)
	var v map[string]any
	d.Decode(&v)

	fmt.Printf("v: %v\n", v)
	for key, value := range v {
		fmt.Printf("key:%v, value:%v\n", key, value)
	}
}
func Json_write_file() {
	f, err := os.OpenFile("test_write.json", os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		log.Panic("Permission Denied")
	}
	defer f.Close()

	e := json.NewEncoder(f)

	p := Person{
		"Alex",
		30,
		"Alex@gmail.com",
		[]string{"Alexander", "Kite"},
	}
	e.Encode(&p)
}
