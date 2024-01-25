from db_config import config_connection
from mysql.connector import connect, Error

# Создаем таблицу для хранения данных пользователей
def db_create_table():
    try:
        with connect(**config_connection) as connection:
            with connection.cursor() as cursor:
                cursor.execute( "CREATE TABLE IF NOT EXISTS users("
                                "   `id` int NOT NULL AUTO_INCREMENT,"
                                "   `user_id` int NOT NULL,"
                                "   `name` varchar(60),"
                                "   `surname` varchar(60),"
                                "   `status_enter` varchar(45) DEFAULT 'out',"
                                "   `photo` blob,"
                                "   `state` varchar(60) DEFAULT 'start',"
                                "   `auth` varchar(60) DEFAULT 'no',"
                                "   `time` float FLOAT NOT NULL DEFAULT '0',"
                                "   PRIMARY KEY (`id`)"
                                ")  ENGINE=InnoDB" )
                connection.commit()
    except Error as e:
        print(e)


# Добавляем id и user_id для пользователя
def db_insert_ids(user_id):
    try:
        with connect(**config_connection) as connection:
            with connection.cursor() as cursor:
                cursor.execute( "INSERT INTO users (`user_id`) VALUES (%s)", (user_id,))
                connection.commit()
    except Error as e:
        print(e)


# Добавляем name и surname для пользователя
def db_set_name(name, user_id):
    try:
        with connect(**config_connection) as connection:
            with connection.cursor() as cursor:
                cursor.execute( "UPDATE users SET `name` = %s WHERE `user_id` = %s", 
                                (name, user_id,))
                connection.commit()
    except Error as e:
        print(e)

def db_set_surname(surname, user_id):
    try:
        with connect(**config_connection) as connection:
            with connection.cursor() as cursor:
                cursor.execute( "UPDATE users SET `surname` = %s WHERE `user_id` = %s", 
                                (surname, user_id,))
                connection.commit()
    except Error as e:
        print(e)

# Получаем name и surname для пользователя
def db_get_names(user_id):
    try:
        with connect(**config_connection) as connection:
            with connection.cursor(buffered=True) as cursor:
                cursor.execute("SELECT name FROM users WHERE user_id = %s", (user_id,))
                results = cursor.fetchone()
                name = str(results[0])
                cursor.execute("SELECT surname FROM users WHERE user_id = %s", (user_id,))
                results = cursor.fetchone()
                surname = str(results[0])
                return [name, surname]
    except Error as e:
        print(e)


# Добавляем status_enter для пользователя
def db_chng_status(user_id):
    cur_status = db_get_status(user_id)
    if cur_status == 'out': 
        new_status = 'in'
    else:
        new_status = 'out'
    try:
        with connect(**config_connection) as connection:
            with connection.cursor() as cursor:
                cursor.execute( "UPDATE users SET `status_enter` = %s WHERE `user_id` = %s", 
                                (new_status, user_id,))
                connection.commit()
    except Error as e:
        print(e)

# Получаем status_enter для пользователя
def db_get_status(user_id):
    try:
        with connect(**config_connection) as connection:
            with connection.cursor(buffered=True) as cursor:
                cursor.execute( "SELECT `status_enter` FROM users WHERE `user_id` = %s", (user_id,))
                result = cursor.fetchone()
                status = str(result[0])
                return status
    except Error as e:
        print(e)


# Добавляем state для пользователя
def db_set_state(state, user_id):
    try:
        with connect(**config_connection) as connection:
            with connection.cursor() as cursor:
                cursor.execute("UPDATE users SET `state` = %s WHERE `user_id` = %s", (state, user_id,))
                connection.commit()
    except Error as e:
        print(e)

# Получаем state для пользователя
def db_get_state(user_id):
    try:
        with connect(**config_connection) as connection:
            with connection.cursor(buffered=True) as cursor:
                cursor.execute("SELECT `state` FROM users WHERE `user_id` = %s", (user_id,))
                result = cursor.fetchone()
                state = str(result[0])
                return state
    except Error as e:
        print(e)

# Добавляем auth для пользователя
def db_set_auth(auth, user_id):
    try:
        with connect(**config_connection) as connection:
            with connection.cursor() as cursor:
                cursor.execute( "UPDATE users SET `auth` = %s WHERE `user_id` = %s", (auth, user_id,))
                connection.commit()
    except Error as e:
        print(e)

# Получаем auth для пользователя
def db_get_auth(user_id):
    try:
        with connect(**config_connection) as connection:
            with connection.cursor(buffered=True) as cursor:
                cursor.execute( "SELECT `auth` FROM users WHERE `user_id` = %s", (user_id,))
                result = cursor.fetchone()
                auth = str(result[0])
                return auth
    except Error as e:
        print(e)

#========================================================================================

# Добавляем photo для пользователя
def convert_to_binary_data(path):
    with open(path, 'rb') as file:
        binaryData = file.read()
    return binaryData

def db_add_photo(path, user_id):
    try:
        with connect(**config_connection) as connection:
            with connection.cursor() as cursor:
                blob_photo = convert_to_binary_data(path)
                cursor.execute("UPDATE users SET `photo` = %s WHERE `user_id` = %s", (blob_photo, user_id,))
                connection.commit()
    except Error as e:
        print(e)

# Получаем photo пользователя
def write_file(binaryData, path, user_id):
    path = f'{path}/{user_id}.jpg'
    with open(path, 'wb') as file:
        file.write(binaryData)

def db_get_photo(path, user_id):
    try:
        with connect(**config_connection) as connection:
            with connection.cursor(buffered=True) as cursor:
                cursor.execute("SELECT `photo` FROM users WHERE `user_id` = %s", (user_id,))
                record = cursor.fetchone()
                image = record[0]
                write_file(image, path, user_id)
    except Error as e:
        print(e)

# Получаем все user_id пользователей
def db_get_userid_list():
    try:
        with connect(**config_connection) as connection:
            with connection.cursor(buffered=True) as cursor:
                cursor.execute("SELECT `user_id` FROM users")
                result = cursor.fetchall()
                userid_list = []
                for res in result:
                    userid_list.append(res[0])
                return userid_list
    except Error as e:
        print(e)

#========================================================================================

# Установить время последнего определения пользователя программой и получить его
def db_set_time(time, user_id):
    try:
        with connect(**config_connection) as connection:
            with connection.cursor() as cursor:
                cursor.execute("UPDATE users SET `time` = %s WHERE `user_id` = %s", (time, user_id,))
                connection.commit()
    except Error as e:
        print(e)

def db_get_time(user_id):
    try:
        with connect(**config_connection) as connection:
            with connection.cursor(buffered=True) as cursor:
                cursor.execute( "SELECT `time` FROM users WHERE `user_id` = %s", (user_id,))
                result = cursor.fetchone()
                time = float(result[0])
                return time
    except Error as e:
        print(e)

#========================================================================================

# Удаляем данные пользователя при остановке
def db_delete_user_data(user_id):
    if (db_user_exists(user_id)):
        try:
            with connect(**config_connection) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("DELETE FROM users WHERE `user_id` = %s", (user_id,))
                    connection.commit()
        except Error as e:
            print(e)
        db_delete_user_data(user_id)

def db_user_exists(user_id):
    try:
        with connect(**config_connection) as connection:
            with connection.cursor(buffered=True) as cursor:
                cursor.execute("SELECT * FROM users WHERE `user_id` = %s", (user_id,))
                result = cursor.fetchall()
                return bool(len(result))
    except Error as e:
        print(e)

if (__name__ == '__main__'):
    db_create_table()
