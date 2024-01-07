import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from fuzzywuzzy import fuzz  # fuzzywuzzy 라이브러리를 사용합니다.

# 사용자로부터 검색어를 입력 받습니다.
search_keyword = input("검색할 제품을 입력하세요: ")

# 검색어를 URL로 안전하게 인코딩합니다.
search_query = quote(search_keyword, safe='')

url = f'https://www.daangn.com/search/{search_query}'  # 검색어를 URL에 포함시킵니다.

# 사이트에 GET 요청을 보내고 응답을 가져옵니다.
response = requests.get(url)

if response.status_code == 200:
    # HTML 페이지를 BeautifulSoup을 사용하여 파싱합니다.
    soup = BeautifulSoup(response.text, 'html.parser')

    # 검색 결과에서 각 제품을 찾습니다.
    products = soup.find_all('article', class_='flea-market-article')

    if not products:
        print('검색 결과가 없습니다.')
    else:
        most_similar_product = None
        max_similarity = 0

        for product in products:
            # 제품명을 추출합니다.
            product_name_elem = product.find('span', class_='article-title')
            product_name = product_name_elem.text.strip() if product_name_elem else '제품명 없음'

            # 가격을 추출합니다.
            product_price_elem = product.find('p', class_='article-price')
            product_price = product_price_elem.text.strip() if product_price_elem else '가격 없음'

            # 입력한 검색어와 제품명 간의 유사성을 측정합니다.
            similarity = fuzz.ratio(search_keyword, product_name)

            # 현재까지 가장 유사한 제품을 업데이트합니다.
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_product = (product_name, product_price)

        # 결과 출력
        if most_similar_product:
            print('-' * 50)
            print(f'가장 근접한 제품 명: {most_similar_product[0]}')
            print(f'가격: {most_similar_product[1]}')
            print('-' * 50)
        else:
            print('가장 근접한 제품을 찾을 수 없습니다.')

else:
    print('검색 결과를 불러올 수 없습니다.')
