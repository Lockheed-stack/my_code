package MysqlTest

import (
	"database/sql"
	"fmt"
	"log"
	"time"

	_ "github.com/go-sql-driver/mysql"
)

// ...

func ConnMysql() *sql.DB {
	// get connection
	db, err := sql.Open("mysql", "root:lilin001@tcp(127.0.0.1:3306)/forGo")
	if err != nil {
		panic(err)
	}
	// See "Important settings" section.
	db.SetConnMaxLifetime(time.Minute * 3)
	db.SetMaxOpenConns(10)
	db.SetMaxIdleConns(10)

	// test connect
	err2 := db.Ping()
	if err2 != nil {
		log.Panic(err2)
	} else {
		fmt.Println("test connection pass")
	}
	return db
}

// insert value
func InsertMysql(db *sql.DB) {
	query_string := "insert into first_table values(?,?)"
	r, err := db.Exec(query_string, 3, "中文")
	if err != nil {
		log.Panic("insert fail")
	} else {
		i, _ := r.LastInsertId()
		fmt.Printf("i: %v\n", i)
	}
}

// select value
func SelectMysqlOneRow(db *sql.DB) {
	type temp struct {
		id   int
		name string
	}
	var t temp
	query_string := "select * from first_table where id=?"
	err := db.QueryRow(query_string, 3).Scan(&t.id, &t.name)
	if err != nil {
		fmt.Printf("err: %v\n", err)
	} else {
		fmt.Printf("t: %v\n", t)
	}
}
func SelectMysqlManyRow(db *sql.DB) {
	type temp struct {
		id   int
		name string
	}
	var t temp

	query_string := "select * from first_table"
	r, err := db.Query(query_string)
	if err != nil {
		log.Panic(err)
	} else {
		for r.Next() {
			r.Scan(&t.id, &t.name)
			fmt.Printf("id: %v,name:%v\n", t.id, t.name)
		}
	}

}

// update value
func UpdateMysqlTable(db *sql.DB) {
	query_string := "update first_table set id=?,name=? where id=?"
	r, err := db.Exec(query_string, 3, "新名字", 3)
	if err != nil {
		log.Panic(err)
	}
	i, _ := r.RowsAffected()
	fmt.Printf("RowsAffected: %v\n", i)
	SelectMysqlManyRow(db)
}
