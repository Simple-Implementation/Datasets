# wikiextractor=3.0.4 다운로드
wget https://github.com/attardi/wikiextractor/archive/refs/tags/v3.0.4.tar.gz
# wikiextractor 압축 해제
tar -xvzf v3.0.4.tar.gz
# tar.gz 삭제
rm v3.0.4.tar.gz
# 디렉토리 이동
cd wikiextractor-3.0.4
# 한국어 위키피디아 최신 덤프 다운로드
wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2
# 한국어 위키피디아 데이터 Parsing
python -m wikiextractor.WikiExtractor kowiki-latest-pages-articles.xml.bz2 
