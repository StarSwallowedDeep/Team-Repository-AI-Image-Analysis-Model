import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import re

# 사용자로부터 검색어를 입력 받습니다.
search_keyword = input("검색할 제품을 입력하세요: ")

# 검색어를 URL로 안전하게 인코딩합니다.
search_query = quote(search_keyword, safe='')

url = f'https://search.shopping.naver.com/search/all?query={search_query}'

response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    products = soup.find_all('div', class_='product_item__MDtDF')

    lowest_price = float('inf')
    lowest_price_product = None

    if not products:
        print('검색 결과가 없습니다.')
    else:
        for product in products:
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
            print(f'최저 가격 제품: {lowest_price_product}')
            print(f'최저 가격: {lowest_price}원')
            print('-' * 50)
        else:
            print('가격 정보를 찾을 수 없습니다.')

else:
    print('검색 결과를 불러올 수 없습니다.')
