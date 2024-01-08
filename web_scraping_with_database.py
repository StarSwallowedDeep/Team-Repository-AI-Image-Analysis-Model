import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from fuzzywuzzy import fuzz
import re
import pymongo

# 함수로 제품 정보 가져오기
def get_productName_from_mongodb():
    # MongoDB에 연결
    client = pymongo.MongoClient("mongodb://localhost:27017")
    
    # 데이터베이스 선택
    db = client["market"]
    
    # 컬렉션 선택
    collection = db["posts"]
    
    # productName 정보를 저장할 리스트
    product_names = []
    
    # MongoDB에서 productName 정보 가져오기    
    results = collection.find({}, {"productName": 1}).sort([("_id", pymongo.DESCENDING)]).limit(1)
    
    for result in results:
        if "productName" in result:
            product_names.append(result["productName"])
            
            global product_id
            product_id = result["_id"]
            
    return product_names, collection

# 함수 호출
product_names_list, collection = get_productName_from_mongodb()

# 반환된 리스트 출력
print(f'검색할 제품 명: {product_names_list[0]}')

# product_names 리스트의 첫 번째 제품명을 사용
search_keyword = product_names_list[0] if product_names_list else "검색어 없음"

# 검색어를 URL로 안전하게 인코딩합니다.
search_query = quote(search_keyword, safe='')

# 네이버 쇼핑 url
naver_shopping_url = f'https://search.shopping.naver.com/search/all?query={search_query}'

# 당근마켓 url
daangn_url = f'https://www.daangn.com/search/{search_query}'

# 네이버 쇼핑 검색 결과 출력
response_naver = requests.get(naver_shopping_url)

if response_naver.status_code == 200:
    soup_naver = BeautifulSoup(response_naver.text, 'html.parser')
    products_naver = soup_naver.find_all('div', class_='product_item__MDtDF')

    lowest_price = float('inf')
    lowest_price_product = None

    if not products_naver:
        print('검색 결과가 없습니다.')
    else:
        for product in products_naver:
            product_name = product.find('a', class_='product_link__TrAac linkAnchor')
            if product_name:
                product_name = product_name.text.strip()
            else:
                product_name = '제품명 없음'

            product_price = product.find('span', class_='price')
            if product_price:
                # 정규표현식을 사용하여 숫자만 추출합니다.
                price_match = re.search(r'\d+', product_price.text.strip().replace(',', ''))
                if price_match:
                    product_price = int(price_match.group())
                    
                    if product_price < lowest_price:
                        lowest_price = product_price
                        lowest_price_product = product_name
                else:
                    continue
            else:
                continue

        if lowest_price_product:
            print('-' * 50)
            print(f'[네이버 쇼핑] 최저 가격 제품: {lowest_price_product}')
            print(f'[네이버 쇼핑] 최저 가격: {lowest_price}원')
            collection.update_one({"_id": product_id}, {"$set": {"cost": lowest_price}})
        else:
            print('가격 정보를 찾을 수 없습니다.')

else:
    print('검색 결과를 불러올 수 없습니다.')   
    
# 당근마켓 검색 결과 출력
response_daangn = requests.get(daangn_url)

if response_daangn.status_code == 200:
    # HTML 페이지를 BeautifulSoup을 사용하여 파싱합니다.
    soup_daangn = BeautifulSoup(response_daangn.text, 'html.parser')

    # 검색 결과에서 각 제품을 찾습니다.
    products_daangn = soup_daangn.find_all('article', class_='flea-market-article')

    if not products_daangn:
        print('검색 결과가 없습니다.')
    else:
        most_similar_product = None
        max_similarity = 0
        
        for product in products_daangn:
            # 제품명을 추출합니다.
            product_name_elem = product.find('span', class_='article-title')
            recent_product = product_name_elem.text.strip() if product_name_elem else '제품명 없음'

            # 가격을 추출합니다.
            product_price_elem = product.find('p', class_='article-price')
            recent_price = product_price_elem.text.strip() if product_price_elem else '가격 없음'
            
            # 입력한 검색어와 제품명 간의 유사성을 측정합니다.
            similarity = fuzz.ratio(search_keyword, recent_product)

            # 현재까지 가장 유사한 제품을 업데이트합니다.
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_product = (recent_product, recent_price)

        # 결과 출력
        if most_similar_product:
            print('-' * 50)
            print(f'[당근마켓] 가장 근접한 제품 명: {most_similar_product[0]}')
            print(f'[당근마켓] 최근 시세 조회: {most_similar_product[1]}')
            print('-' * 50)
            collection.update_one({"_id": product_id}, {"$set": {"recentPrice": most_similar_product[1]}})
        else:
            print('가장 근접한 제품을 찾을 수 없습니다.')
else:
    print('당근마켓 검색 결과를 불러올 수 없습니다.')
